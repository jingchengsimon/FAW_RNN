"""
Standalone RNN Sector training script
Used to train RNN models and save results
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== Acceleration Training Modules (Optional) ====================
# These modules are only used when use_acceleration=True
# By default (use_acceleration=False), they are not imported to keep the code clean

def _init_acceleration_modules():
    """Initialize acceleration training related modules (imported only when needed)"""
    try:
        from torch.amp import autocast, GradScaler
        try:
            import psutil
        except ImportError:
            psutil = None
        return autocast, GradScaler, psutil
    except ImportError:
        return None, None, None

def _get_gpu_memory_usage():
    """Get current GPU memory usage (only used in acceleration mode)"""
    if not torch.cuda.is_available():
        return 0.0
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    return (reserved / total) * 100.0 if total > 0 else 0.0

def _find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256):
    """
    Automatically find optimal batch_size (only used in acceleration mode)
    
    Args:
        model: Model
        train_data: Training dataset
        device: Device
        start_batch_size: Starting batch_size
        max_batch_size: Maximum batch_size
    
    Returns:
        Optimal batch_size
    """
    if device == 'cpu':
        return start_batch_size
    
    model.eval()
    optimal_batch_size = start_batch_size
    
    # Test different batch sizes
    for batch_size in [start_batch_size, 64, 128, 256]:
        if batch_size > max_batch_size:
            break
        
        try:
            torch.cuda.empty_cache()
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_batch = next(iter(test_loader))
            inputs, labels = test_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                _ = model(inputs)
            
            memory_usage = _get_gpu_memory_usage()
            
            if memory_usage < 80.0:
                optimal_batch_size = batch_size
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, usable")
            else:
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, exceeds limit")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} caused OOM, using batch_size={optimal_batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    torch.cuda.empty_cache()
    model.train()
    return optimal_batch_size


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
        """
        self.data = data
        self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors

    def __len__(self):
        return (self.data.shape[0]-self.chan_num) // self.frame_num

    def __getitem__(self, idx):
        start_idx = (idx * self.frame_num) + self.chan_num
        end_idx = start_idx + self.frame_num

        # Stack frames to create a multichannel image
        for i in range(-(self.chan_num-1), 1):
            if i == -(self.chan_num-1):
                stacked_frames = np.expand_dims(self.data[(start_idx + i):(end_idx + i)], axis=1)
            else:
                stacked_frames = np.concatenate((stacked_frames,
                                                 np.expand_dims(self.data[(start_idx + i):(end_idx + i)],
                                                                axis=1)), axis=1)
        stacked_frames = stacked_frames.astype(np.float32)

        # labels: (frame_num, 3) -> [char_id, x, y]
        labels = self.labels[start_idx:end_idx].copy()

        if self.use_sector:
            # Use image width and height to map (x, y) to a grid_size x grid_size grid,
            # obtaining sector id 0-(num_sectors-1) (e.g., num_sectors=9 -> 3x3 grid)
            height = self.data.shape[-2]
            width = self.data.shape[-1]

            # Derive grid_size for each dimension from num_sectors (assuming num_sectors is a perfect square, e.g., 9, 16)
            grid_size = int(np.sqrt(self.num_sectors))
            if grid_size * grid_size != self.num_sectors:
                raise ValueError(f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid")

            x = labels[:, 1].astype(np.float32)
            y = labels[:, 2].astype(np.float32)

            # 将坐标归一化到 [0, grid_size) 再取整，注意用 (width-1)/(height-1) 避免越界
            col = (x / max(width - 1, 1) * grid_size).astype(np.int64)
            row = (y / max(height - 1, 1) * grid_size).astype(np.int64)

            # 防止因为数值或边界问题越界
            col = np.clip(col, 0, grid_size - 1)
            row = np.clip(row, 0, grid_size - 1)

            # 按行优先编码 sector id: row * grid_size + col，范围 0-(num_sectors-1)
            sector = row * grid_size + col

            # 新的 label: [char_id, sector_id]
            labels = np.stack([labels[:, 0].astype(np.int64), sector], axis=1)

        return stacked_frames, labels


# ==================== Model Classes ====================
class RNNConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3):
        super(RNNConv, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        # Original RNN version
        self.rnn = nn.RNN(input_size=64 * 12 * 12, hidden_size=256,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(256)
        self.fcchar = nn.Linear(256, num_classes)
        self.fcpos = nn.Linear(256, num_pos)
        self.to(self.device)

    def encoder(self, x):
        # Add dropout to prevent overfitting
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # 2D dropout for conv layers
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)
        
        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class GaWFRNNConv(nn.Module):
    """
    GaWF (Gated with Feedback) RNN Model
    
    Main improvements:
    1. Use classifier output as feedback to RNN input
    2. Feedback is transformed by U @ diag(concat) @ V, then Hadamard product with RNN weights
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3):
        super(GaWFRNNConv, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.dropout_rate = dropout_rate
        
        # CNN encoder (same as RNNConv)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        
        # RNN parameters
        input_size = 64 * 12 * 12  # 9216
        hidden_size = 256
        
        # Create RNN (but not using built-in, manually implement to support feedback)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        
        # Feedback transformation matrices
        # Dimension after concatenating classifier outputs
        feedback_dim = num_classes + num_pos  # e.g., 10 + 9 = 19
        
        # RNN weight matrix shapes
        # weight_ih: (hidden_size, input_size) = (256, 9216)
        # weight_hh: (hidden_size, hidden_size) = (256, 256)
        # Concatenated shape: (256, 9216 + 256) = (256, 9472)
        combined_weight_size = input_size + hidden_size  # 9472
        
        # U: (hidden_size, feedback_dim) = (256, 19)
        # V: (feedback_dim, combined_weight_size) = (19, 9472)
        # diag(concat): (feedback_dim, feedback_dim) = (19, 19)
        # U @ diag @ V: (256, 19) @ (19, 19) @ (19, 9472) = (256, 9472)
        self.U = nn.Parameter(torch.randn(hidden_size, feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feedback_dim, combined_weight_size) * 0.01)
        
        # LayerNorm and Dropout
        self.LNormRNN = nn.LayerNorm(hidden_size)
        
        # Classifier heads
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        
        # Store previous forward's classifier output as feedback for next time
        self.register_buffer('prev_feedback', None)
        
        self.to(self.device)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x, feedback=None):
        """
        GaWF RNN middle layer with feedback mechanism
        
        According to the diagram (Panel A):
        1. Concatenate Input and Feedback
        2. Matrix multiplication with weight matrix (standard RNN computation)
        3. Feedback transformed by U @ diag(concat) @ V, then sigmoid (gating signal)
        4. Hadamard product of the two results (element-wise multiplication)
        5. LayerNorm
        6. ReLU
        
        Args:
            x: Input sequence (B, T, input_size)
            feedback: Feedback from classifier (B, T, feedback_dim) or None
        """
        batch_size, seq_len, input_size = x.size()
        hidden_size = self.rnn.hidden_size
        
        if feedback is not None:
            # Get RNN weights
            weight_ih = self.rnn.weight_ih_l0  # (hidden_size, input_size)
            weight_hh = self.rnn.weight_hh_l0  # (hidden_size, hidden_size)
            bias_ih = self.rnn.bias_ih_l0 if self.rnn.bias_ih_l0 is not None else None
            bias_hh = self.rnn.bias_hh_l0 if self.rnn.bias_hh_l0 is not None else None
            
            # Process each time step
            outputs = []
            h = torch.zeros(batch_size, hidden_size, device=x.device)  # (B, hidden_size)
            
            # Concatenate RNN input weight matrix and recurrent weight matrix (compute once)
            combined_weight = torch.cat([weight_ih, weight_hh], dim=1)  # (hidden_size, input_size + hidden_size)
            
            for t in range(seq_len):
                # Current time step input and feedback
                x_t = x[:, t, :]  # (B, input_size)
                fb_t = feedback[:, t, :]  # (B, feedback_dim)
                
                # Feedback transformation: U @ diag(fb_t) @ V, then sigmoid
                # fb_t: (B, feedback_dim)
                # For each sample b, compute U @ diag(fb_t[b]) @ V
                # Equivalent to: U @ (fb_t[b] * I) @ V, where I is identity matrix
                # Can be written as: U @ (fb_t[b].unsqueeze(1) * V), but this is incorrect
                # Correct way: For diag(fb_t[b]), fb_t[b] serves as diagonal elements
                # U @ diag(fb_t[b]) @ V = U @ (torch.diag(fb_t[b])) @ V
                # Can be vectorized: use batch matrix multiplication
                
                # Method: Build diagonal matrix for each sample separately and compute (current implementation)
                # But can use more efficient vectorization
                gated_weights = []
                for b in range(batch_size):
                    # Build diagonal matrix: diag(fb_t[b])
                    diag_matrix = torch.diag(fb_t[b])  # (feedback_dim, feedback_dim)
                    
                    # U @ diag @ V: (hidden_size, combined_weight_size)
                    transformed = self.U @ diag_matrix @ self.V  # (hidden_size, combined_weight_size)
                    
                    # Apply sigmoid
                    transformed = torch.sigmoid(transformed)  # (hidden_size, combined_weight_size)
                    
                    # Hadamard product: transformed * combined_weight
                    gated_weight = transformed * combined_weight  # (hidden_size, combined_weight_size)
                    gated_weights.append(gated_weight)
                
                # Stack gated weights: (B, hidden_size, combined_weight_size)
                gated_weights = torch.stack(gated_weights, dim=0)  # (B, hidden_size, combined_weight_size)
                
                # Separate back into weight_ih and weight_hh parts
                gated_weight_ih = gated_weights[:, :, :input_size]  # (B, hidden_size, input_size)
                gated_weight_hh = gated_weights[:, :, input_size:]  # (B, hidden_size, hidden_size)
                
                # Compute RNN output using gated weights
                # Compute for each sample separately
                h_t_list = []
                for b in range(batch_size):
                    # Input to hidden: (1, hidden_size)
                    ih = F.linear(x_t[b:b+1], gated_weight_ih[b], bias_ih)  # (1, hidden_size)
                    # Hidden to hidden: (1, hidden_size)
                    hh = F.linear(h[b:b+1], gated_weight_hh[b], bias_hh)  # (1, hidden_size)
                    # Combine and apply activation
                    h_t = torch.tanh(ih + hh)  # (1, hidden_size)
                    h_t_list.append(h_t)
                
                # Stack: (B, hidden_size)
                h_t = torch.cat(h_t_list, dim=0)  # (B, hidden_size)
                
                # LayerNorm and ReLU
                gated_output = self.LNormRNN(h_t)  # (B, hidden_size)
                gated_output = F.relu(gated_output)  # (B, hidden_size)
                
                outputs.append(gated_output)
                h = gated_output  # Update hidden state
            
            # Stack outputs: (B, T, hidden_size)
            x = torch.stack(outputs, dim=1)  # (B, T, hidden_size)
        else:
            # When no feedback, use standard RNN
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
        
        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x, use_feedback=True, reset_feedback=False):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
            use_feedback: Whether to use feedback mechanism
            reset_feedback: Whether to reset feedback (for new sequence start, e.g., new epoch or new batch)
        """
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)
        
        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # Determine whether to use feedback
        if use_feedback:
            # If reset_feedback is True, or prev_feedback is None (first forward)
            if reset_feedback or self.prev_feedback is None:
                # First forward: don't use feedback
                feedback = None
            else:
                # Use previous forward's saved classifier output as feedback
                feedback = self.prev_feedback  # (B, T, feedback_dim)
        else:
            feedback = None
        
        # appl RNN
        x = self.middle(x, feedback=feedback)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        
        # If using feedback, save current output as feedback for next time
        if use_feedback:
            # Concatenate two classifier outputs
            # char_out: (B, T, num_classes)
            # pos_out: (B, T, num_pos)
            # prev_feedback: (B, T, num_classes + num_pos)
            # Use .detach() to break gradient connection, avoid computation graph issues
            self.prev_feedback = torch.cat([char_out, pos_out], dim=-1).detach()  # (B, T, feedback_dim)
            
        return char_out, pos_out


class GRUConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3):
        super(GRUConv, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = nn.GRU(input_size=64 * 12 * 12, hidden_size=256,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(256)
        self.fcchar = nn.Linear(256, num_classes)
        self.fcpos = nn.Linear(256, num_pos)
        self.to(self.device)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)

        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class LSTMConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3):
        super(LSTMConv, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = nn.LSTM(input_size=64 * 12 * 12, hidden_size=256,
                           num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(256)
        self.fcchar = nn.Linear(256, num_classes)
        self.fcpos = nn.Linear(256, num_pos)
        self.to(self.device)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)

        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


# ==================== Training Function ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                  use_acceleration=False, weight_decay=1e-4, dropout_rate=0.5, 
                  early_stopping_patience=15, min_delta=0.001):
    """
    Train model, supports sector mode and coordinate mode
    
    Args:
        mdl: Model
        train_data: Training dataset (MC_RNN_Dataset)
        val_data: Validation dataset (MC_RNN_Dataset)
        num_epochs: Number of training epochs
        loss_weights: [character loss weight, position loss weight], if None, automatically set based on use_sector
                     - sector mode default: [1, 1]
                     - coordinate mode default: [1, 0.001]
        lr: Learning rate
        use_acceleration: Whether to use acceleration training (default False)
                         - True: Enable mixed precision training, automatic batch_size optimization, DataLoader optimization, etc.
                         - False: Use original training method, does not affect existing logic
        weight_decay: L2 regularization coefficient (weight decay), default 1e-4, used to prevent overfitting
        dropout_rate: Dropout rate, default 0.5 (note: need to manually modify dropout values in model)
        early_stopping_patience: Early stopping patience, number of epochs without validation improvement, default 15
        min_delta: Minimum improvement threshold for early stopping, default 0.001
    """
    # Get use_sector information from dataset
    use_sector = train_data.use_sector
    
    # Set default loss_weights based on use_sector
    if loss_weights is None:
        if use_sector:
            loss_weights = [1, 1]  # sector mode: character and sector loss weights equal
        else:
            loss_weights = [1, 0.001]  # coordinate mode: position loss weight smaller (MSE usually has larger values)
    
    # Place parameters according to model's internal device (can be 'cuda' or 'cpu')
    device = mdl.device
    mdl.to(device)
    
    # ========== Acceleration Training Module Initialization (only used when use_acceleration=True) ==========
    autocast_fn = None
    GradScaler_cls = None
    scaler = None
    psutil_module = None
    batch_size = 32  # Default batch_size
    num_workers = 0   # Default single process
    pin_memory = False  # Default not using pin_memory
    show_gpu_usage = False  # Default not showing GPU usage
    
    if use_acceleration:
        print("Enabling acceleration training...")
        autocast_fn, GradScaler_cls, psutil_module = _init_acceleration_modules()
        
        if autocast_fn is None or GradScaler_cls is None:
            print("Warning: Unable to import acceleration training modules, will use standard training")
            use_acceleration = False
        else:
            # Automatically find optimal batch_size
            # Note: GaWFRNNConv model skips batch_size search, uses default value 32
            # Because GaWFRNNConv uses feedback mechanism, batch_size changes cause prev_feedback dimension mismatch
            if device == 'cuda' and not isinstance(mdl, GaWFRNNConv):
                print("Automatically finding optimal batch_size...")
                batch_size = _find_optimal_batch_size(mdl, train_data, device=device)
                print(f"Using batch_size = {batch_size}")
            elif isinstance(mdl, GaWFRNNConv):
                print(f"Detected GaWFRNNConv model, skipping batch_size search, using default batch_size = {batch_size}")
            
            # Automatically set num_workers
            if psutil_module is not None:
                num_workers = min(4, psutil_module.cpu_count(logical=False))
            else:
                import os
                num_workers = min(4, os.cpu_count() or 1)
            
            # Enable pin_memory (GPU only)
            pin_memory = (device == 'cuda')
            show_gpu_usage = True
            
            # Initialize mixed precision training scaler
            if device == 'cuda':
                scaler = GradScaler_cls('cuda')
                print(f"Acceleration settings: batch_size={batch_size}, num_workers={num_workers}, "
                      f"pin_memory={pin_memory}, mixed precision training=enabled")
            
            # In acceleration mode, limit batch_size to no more than 32 to maintain same convergence characteristics as original mode
            # Larger batch_size changes gradient estimation characteristics, may slow convergence
            # Acceleration mainly achieved through mixed precision training, not by increasing batch_size
            if batch_size > 32:
                original_batch_size = batch_size
                batch_size = 32
                print(f"batch_size limited from {original_batch_size} to {batch_size} to maintain same convergence speed as original mode")
    # ========== Acceleration Module Initialization End ==========
    
    # Add weight decay (L2 regularization) to prevent overfitting
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_char = nn.CrossEntropyLoss()
    
    # Learning rate hard decay settings: decay learning rate at specific epochs
    # Decay points: at 25%, 50%, 75% of total epochs, decay to 0.5x original
    lr_decay_epochs = [int(num_epochs * 0.25), int(num_epochs * 0.5), int(num_epochs * 0.75)]
    lr_decay_factor = 0.5  # Decay to 0.5x each time
    
    # Learning rate warmup settings: only used in acceleration mode
    # Gradually increase learning rate in early training, helps stability of mixed precision training
    use_warmup = use_acceleration and device == 'cuda'
    warmup_epochs = 5 if use_warmup else 0  # First 5 epochs for warmup
    initial_lr = lr  # Save initial learning rate
    
    if use_warmup:
        print(f"Learning rate warmup settings: first {warmup_epochs} epochs from {lr * 0.1:.6f} linearly increase to {lr:.6f}")
    print(f"Learning rate decay settings: at epochs {lr_decay_epochs}, decay learning rate to {lr_decay_factor}x original")
    
    # Select position loss function based on use_sector
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss()  # sector classification
    else:
        criterion_pos = nn.MSELoss()  # coordinate regression
    
    def loss_fn(out_char, out_pos, labels):
        # Character loss (same for both modes)
        labels_char = labels[:, :, 0].long().view(-1)
        outputs_char = out_char.view(-1, out_char.shape[-1])  # (B*T, num_classes)
        loss_char = criterion_char(outputs_char, labels_char)
        
        # Position loss (different methods based on use_sector)
        if use_sector:
            # sector mode: classification loss
            labels_pos = labels[:, :, 1].long().view(-1)
            outputs_pos = out_pos.view(-1, out_pos.shape[-1])  # (B*T, num_sectors)
            loss_pos = criterion_pos(outputs_pos, labels_pos)
        else:
            # coordinate mode: regression loss (MSE)
            labels_pos = labels[:, :, 1:].float()  # (B, T, 2) -> [x, y]
            outputs_pos = out_pos  # (B, T, 2) -> [x, y]
            loss_pos = criterion_pos(outputs_pos, labels_pos)

        # Keep regularization consistent with original (if model doesn't have mdl.rnn, need corresponding modification)
        rnn_hh = mdl.rnn.weight_hh_l0
        rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
        loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_metric_pos = 0  # sector mode: accuracy; coordinate mode: MSE
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                
                # Reset feedback at start of each batch (if GaWFRNNConv)
                # This ensures feedback's batch_size and seq_len match current batch
                if hasattr(mdl, 'prev_feedback'):
                    mdl.prev_feedback = None
                
                # Select data transfer method based on acceleration mode
                if use_acceleration and pin_memory:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    labels = labels.to(device)
                
                # Select whether to use mixed precision based on acceleration mode
                if use_acceleration and scaler is not None and autocast_fn is not None:
                    with autocast_fn('cuda'):
                        out_char, out_pos = mdl(inputs)
                else:
                    out_char, out_pos = mdl(inputs)
                
                # char 精度（两种模式相同）
                total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                
                # 位置相关指标（根据 use_sector 选择不同方式）
                if use_sector:
                    # sector 模式：计算准确率
                    total_metric_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
                else:
                    # 坐标模式：计算 MSE
                    labels_pos = labels[:, :, 1:].float()  # (B, T, 2)
                    total_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
        
        # Return results
        acc_char = total_acc_char * 100 / len(data_loader)
        if use_sector:
            metric_pos = total_metric_pos * 100 / len(data_loader)  # accuracy (percentage)
        else:
            metric_pos = total_metric_pos / len(data_loader)  # MSE (pixel squared)
        return acc_char, metric_pos

    # data loader (select different configurations based on acceleration mode)
    if use_acceleration:
        train_dl = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        val_dl = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,  # validation set doesn't need shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        # Original method: simple configuration
        train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=32, shuffle=True)
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)  # sector mode: accuracy; coordinate mode: MSE
    val_metric_pos = np.zeros(num_epochs)
    
    # Early stopping mechanism
    best_val_metric = -np.inf if use_sector else np.inf  # sector mode: larger is better; coordinate mode: smaller is better
    best_val_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Learning rate warmup: gradually increase learning rate in early training (only in acceleration mode)
        if use_warmup and epoch < warmup_epochs:
            # Linear warmup: linearly increase from initial_lr * 0.1 to initial_lr
            warmup_lr = initial_lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
            for param_group in optim.param_groups:
                param_group['lr'] = warmup_lr
            if epoch == 0 or epoch == warmup_epochs - 1:
                print(f"Epoch {epoch + 1}: Learning rate warmed up to {warmup_lr:.6f}")
        # Learning rate hard decay: decay learning rate at specified epochs
        elif epoch in lr_decay_epochs:
            current_lr = optim.param_groups[0]['lr']
            new_lr = current_lr * lr_decay_factor
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr
            print(f"Epoch {epoch + 1}: Learning rate decayed from {current_lr:.6f} to {new_lr:.6f}")
        
        mdl.train()
        
        # Training loop
        epoch_train_acc_char = 0.0
        epoch_train_metric_pos = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dl):
            inputs, labels = batch
            
            # Reset feedback at start of each batch (if GaWFRNNConv)
            # This ensures feedback's batch_size and seq_len match current batch
            if hasattr(mdl, 'prev_feedback'):
                mdl.prev_feedback = None
            
            # Select data transfer method based on acceleration mode
            if use_acceleration and pin_memory:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                labels = labels.to(device)
            
            optim.zero_grad()
            
            # Select whether to use mixed precision training based on acceleration mode
            if use_acceleration and scaler is not None and autocast_fn is not None:
                with autocast_fn('cuda'):
                    out_char, out_pos = mdl(inputs)
                    loss = loss_fn(out_char, out_pos, labels)
                
                scaler.scale(loss).backward()
                # Gradient clipping (mixed precision training)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                scaler.step(optim)
                scaler.update()
            else:
                # Original method: standard training
                out_char, out_pos = mdl(inputs)
                loss = loss_fn(out_char, out_pos, labels)
                loss.backward()
                # Gradient clipping (standard training)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                optim.step()
            
            # 计算训练指标
            # 字符准确率（两种模式相同）
            epoch_train_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            
            # 位置相关指标（根据 use_sector 选择不同方式）
            if use_sector:
                # sector 模式：计算准确率
                epoch_train_metric_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
            else:
                # 坐标模式：计算 MSE
                labels_pos = labels[:, :, 1:].float()  # (B, T, 2)
                epoch_train_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
            
            num_batches += 1
            
            # Acceleration mode: periodically clear GPU cache
            if use_acceleration and batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate epoch average metrics
        train_acc_char[epoch] = (epoch_train_acc_char / num_batches) * 100
        if use_sector:
            train_metric_pos[epoch] = (epoch_train_metric_pos / num_batches) * 100  # accuracy (percentage)
        else:
            train_metric_pos[epoch] = epoch_train_metric_pos / num_batches  # MSE (pixel squared)
        
        # Format output string
        gpu_info = ""
        if use_acceleration and show_gpu_usage and device == 'cuda' and torch.cuda.is_available():
            gpu_mem = _get_gpu_memory_usage()
            gpu_info = f" | GPU memory: {gpu_mem:.1f}%"
        
        if use_sector:
            train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f}%){gpu_info}"
        else:
            train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f} pix^2){gpu_info}"

        with torch.no_grad():
            val_acc_char[epoch], val_metric_pos[epoch] = evaluate(mdl, val_dl)
            if use_sector:
                val_str = f" Validation (char, sector): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f}%)"
            else:
                val_str = f" Validation (char, pos): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f} pix^2)"
            print(train_str, val_str)
            
            # Early stopping check: judge based on validation accuracy of main task (character recognition)
            current_val_metric = val_acc_char[epoch]
            if use_sector:
                # sector mode: use weighted average of character accuracy and sector accuracy
                current_val_metric = (val_acc_char[epoch] + val_metric_pos[epoch]) / 2.0
            
            improved = False
            if use_sector:
                # sector mode: higher accuracy is better
                if current_val_metric > best_val_metric + min_delta:
                    improved = True
            else:
                # coordinate mode: smaller MSE is better (but here use character accuracy as main metric)
                if current_val_metric > best_val_metric + min_delta:
                    improved = True
            
            if improved:
                best_val_metric = current_val_metric
                best_val_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = mdl.state_dict().copy()
                print(f"  ✓ Validation performance improved! Current best: {best_val_metric:.2f} (epoch {epoch + 1})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered: validation performance did not improve for {early_stopping_patience} epochs")
                    print(f"Best validation performance: {best_val_metric:.2f} (epoch {best_val_epoch + 1})")
                    # Restore best model
                    if best_model_state is not None:
                        mdl.load_state_dict(best_model_state)
                        print("Best model state restored")
                    break

    torch.cuda.empty_cache()

    # If early stopping triggered, only return actual trained epochs (epoch starts from 0, so actually trained epoch+1 epochs)
    actual_epochs = epoch + 1
    
    # Return different key names based on use_sector, only return actual trained epochs
    if use_sector:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_acc_pos": train_metric_pos[:actual_epochs],  # sector accuracy
            "val_acc_pos": val_metric_pos[:actual_epochs],      # sector accuracy
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # save actual trained epochs
        }
    else:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_err_pos": train_metric_pos[:actual_epochs],  # coordinate MSE
            "val_err_pos": val_metric_pos[:actual_epochs],      # coordinate MSE
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # save actual trained epochs
        }


# ==================== Utility Functions ====================
def save_results(results, filepath):
    """
    Save training results to local file
    
    Args:
        results: Training results dictionary
        filepath: Save path (e.g., 'results_rnn' or 'results_rnn_sector')
    """
    # Create save dictionary (does not include model, because model is too large)
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # Save model state dict (can be saved separately if needed)
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # Verify model's num_pos before saving
        saved_num_pos = results["model"].fcpos.out_features
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # Save other results
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


# ==================== Main Training Code ====================
if __name__ == "__main__":
    # Data path configuration
    stim_train_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-train.npy"
    label_train_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-train.tsv"
    stim_val_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-validation.npy"
    label_val_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-validation.tsv"
    
    # 加载数据
    print("Loading data...")
    stims_train = np.load(stim_train_path, allow_pickle=True)
    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    
    stims_val = np.load(stim_val_path, allow_pickle=True)
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    
    # Create datasets (can choose sector mode or coordinate mode)
    print("Creating datasets...")
    use_sector_mode = True   # Set to True to use sector mode, False to use coordinate mode
    use_acceleration = True  # Set to True to enable acceleration training, False to use original method
    
    # Select model type: 'rnn' (RNNConv), 'lstm' (LSTMConv), 'gru' (GRUConv), 'gawf' (GaWFRNNConv)
    model_type = "gawf"
    
    if use_sector_mode:
        # sector mode: 3x3 grid -> 9 sectors
        num_pos = 9  # number of sectors
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True, num_sectors=num_pos)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True, num_sectors=num_pos)
        print("Using sector mode (3x3 grid, 9 sectors)")
    else:
        # coordinate mode: directly predict (x, y) coordinates
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=False)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=False)
        num_pos = 2  # x, y coordinates
        print("Using coordinate mode (directly predict x, y coordinates)")
    
    # Create model (position output num_pos set according to mode)
    print("Creating model...")
    # Model class mapping table
    MODEL_CLASSES = {
        "rnn": RNNConv,
        "lstm": LSTMConv,
        "gru": GRUConv,
        "gawf": GaWFRNNConv,
    }
    
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected one of {list(MODEL_CLASSES.keys())}")
    
    ModelClass = MODEL_CLASSES[model_type]
    dropout_rate = 0.3  # CNN encoder的dropout率
    mdl_rnn = ModelClass(num_classes=10, num_pos=num_pos, kernel_size=5, dropout_rate=dropout_rate)
    model_suffix = model_type
    print(f"使用模型类型: {model_type.upper()} (num_pos={num_pos}, dropout_rate={dropout_rate})")
    
    # 训练模型（loss_weights 会根据 use_sector 自动设置，也可以手动指定）
    print("Starting training...")
    
    if use_acceleration:
        print("加速训练功能已启用")
    else:
        print("使用标准训练方式")
    
    results_rnn = network_train(
        mdl_rnn, 
        train_ds, 
        val_ds, 
        num_epochs=200, 
        use_acceleration=use_acceleration,  # 控制是否使用加速训练
        weight_decay=1e-4,  # L2正则化，防止过拟合
        dropout_rate=0.3,  # CNN encoder的dropout率
        early_stopping_patience=15,  # 早停耐心值
        min_delta=0.001  # 早停最小改进阈值
    )
    
    # 保存训练结果
    print("\nSaving results...")
    mode_suffix = "sector" if use_sector_mode else "coord"
    acc_suffix = "_acc" if use_acceleration else ""
    # 文件命名同时包含模式和模型类型，便于区分不同实验
    # 例如：results_rnn_sector.pkl, results_lstm_coord_acc.pkl
    results_rnn_path = f"results_{model_suffix}_{mode_suffix}{acc_suffix}_2"
    
    save_results(results_rnn, results_rnn_path)
    
    print("\nTraining completed!")

