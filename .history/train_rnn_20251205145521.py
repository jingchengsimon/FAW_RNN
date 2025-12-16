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
        
        # 创建 RNN（但不使用内置的，而是手动实现以支持 feedback）
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        
        # Feedback 变换矩阵
        # classifier 输出 concat 后的维度
        feedback_dim = num_classes + num_pos  # 例如：10 + 9 = 19
        
        # RNN 权重矩阵的形状
        # weight_ih: (hidden_size, input_size) = (256, 9216)
        # weight_hh: (hidden_size, hidden_size) = (256, 256)
        # 拼接后的形状：(256, 9216 + 256) = (256, 9472)
        combined_weight_size = input_size + hidden_size  # 9472
        
        # U: (hidden_size, feedback_dim) = (256, 19)
        # V: (feedback_dim, combined_weight_size) = (19, 9472)
        # diag(concat): (feedback_dim, feedback_dim) = (19, 19)
        # U @ diag @ V: (256, 19) @ (19, 19) @ (19, 9472) = (256, 9472)
        self.U = nn.Parameter(torch.randn(hidden_size, feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feedback_dim, combined_weight_size) * 0.01)
        
        # LayerNorm 和 Dropout
        self.LNormRNN = nn.LayerNorm(hidden_size)
        
        # Classifier heads
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        
        # 存储上一次 forward 的 classifier 输出，作为下一次的 feedback
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
        
        根据图的示意（Panel A）：
        1. Input 和 Feedback concat
        2. 与权重矩阵做 matmul（标准 RNN 计算）
        3. Feedback 经过 U @ diag(concat) @ V，然后 sigmoid（门控信号）
        4. 两个结果做 Hadamard product（逐元素相乘）
        5. LayerNorm
        6. ReLU
        
        Args:
            x: 输入序列 (B, T, input_size)
            feedback: 来自 classifier 的 feedback (B, T, feedback_dim) 或 None
        """
        batch_size, seq_len, input_size = x.size()
        hidden_size = self.rnn.hidden_size
        
        if feedback is not None:
            # 获取 RNN 权重
            weight_ih = self.rnn.weight_ih_l0  # (hidden_size, input_size)
            weight_hh = self.rnn.weight_hh_l0  # (hidden_size, hidden_size)
            bias_ih = self.rnn.bias_ih_l0 if self.rnn.bias_ih_l0 is not None else None
            bias_hh = self.rnn.bias_hh_l0 if self.rnn.bias_hh_l0 is not None else None
            
            # 对每个时间步进行处理
            outputs = []
            h = torch.zeros(batch_size, hidden_size, device=x.device)  # (B, hidden_size)
            
            # 拼接 RNN 的输入权重矩阵和 recurrent 权重矩阵（只需要计算一次）
            combined_weight = torch.cat([weight_ih, weight_hh], dim=1)  # (hidden_size, input_size + hidden_size)
            
            for t in range(seq_len):
                # 当前时间步的输入和 feedback
                x_t = x[:, t, :]  # (B, input_size)
                fb_t = feedback[:, t, :]  # (B, feedback_dim)
                
                # Feedback 变换：U @ diag(fb_t) @ V，然后 sigmoid
                # fb_t: (B, feedback_dim)
                # 对每个样本 b，计算 U @ diag(fb_t[b]) @ V
                # 等价于：U @ (fb_t[b] * I) @ V，其中 I 是单位矩阵
                # 可以写成：U @ (fb_t[b].unsqueeze(1) * V)，但这样不对
                # 正确的方式：对于 diag(fb_t[b])，相当于 fb_t[b] 作为对角元素
                # U @ diag(fb_t[b]) @ V = U @ (torch.diag(fb_t[b])) @ V
                # 可以向量化为：使用批量矩阵乘法
                
                # 方法：对每个样本分别构建对角矩阵并计算（当前实现）
                # 但可以使用更高效的向量化方式
                gated_weights = []
                for b in range(batch_size):
                    # 构建对角矩阵：diag(fb_t[b])
                    diag_matrix = torch.diag(fb_t[b])  # (feedback_dim, feedback_dim)
                    
                    # U @ diag @ V: (hidden_size, combined_weight_size)
                    transformed = self.U @ diag_matrix @ self.V  # (hidden_size, combined_weight_size)
                    
                    # 应用 sigmoid
                    transformed = torch.sigmoid(transformed)  # (hidden_size, combined_weight_size)
                    
                    # Hadamard product: transformed * combined_weight
                    gated_weight = transformed * combined_weight  # (hidden_size, combined_weight_size)
                    gated_weights.append(gated_weight)
                
                # Stack gated weights: (B, hidden_size, combined_weight_size)
                gated_weights = torch.stack(gated_weights, dim=0)  # (B, hidden_size, combined_weight_size)
                
                # 分离回 weight_ih 和 weight_hh 的部分
                gated_weight_ih = gated_weights[:, :, :input_size]  # (B, hidden_size, input_size)
                gated_weight_hh = gated_weights[:, :, input_size:]  # (B, hidden_size, hidden_size)
                
                # 使用门控后的权重计算 RNN 输出
                # 对每个样本分别计算
                h_t_list = []
                for b in range(batch_size):
                    # 输入到隐藏: (1, hidden_size)
                    ih = F.linear(x_t[b:b+1], gated_weight_ih[b], bias_ih)  # (1, hidden_size)
                    # 隐藏到隐藏: (1, hidden_size)
                    hh = F.linear(h[b:b+1], gated_weight_hh[b], bias_hh)  # (1, hidden_size)
                    # 组合并应用激活
                    h_t = torch.tanh(ih + hh)  # (1, hidden_size)
                    h_t_list.append(h_t)
                
                # Stack: (B, hidden_size)
                h_t = torch.cat(h_t_list, dim=0)  # (B, hidden_size)
                
                # LayerNorm 和 ReLU
                gated_output = self.LNormRNN(h_t)  # (B, hidden_size)
                gated_output = F.relu(gated_output)  # (B, hidden_size)
                
                outputs.append(gated_output)
                h = gated_output  # 更新隐藏状态
            
            # Stack outputs: (B, T, hidden_size)
            x = torch.stack(outputs, dim=1)  # (B, T, hidden_size)
        else:
            # 没有 feedback 时，使用标准 RNN
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
            x: 输入序列 (B, T, C, H, W)
            use_feedback: 是否使用 feedback 机制
            reset_feedback: 是否重置 feedback（用于新序列开始时，如新的 epoch 或新的 batch）
        """
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)
        
        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # 确定是否使用 feedback
        if use_feedback:
            # 如果 reset_feedback 为 True，或者 prev_feedback 为 None（第一次 forward）
            if reset_feedback or self.prev_feedback is None:
                # 第一次 forward：不使用 feedback
                feedback = None
            else:
                # 使用上一次 forward 保存的 classifier 输出作为 feedback
                feedback = self.prev_feedback  # (B, T, feedback_dim)
        else:
            feedback = None
        
        # appl RNN
        x = self.middle(x, feedback=feedback)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        
        # 如果使用 feedback，保存当前输出作为下一次的 feedback
        if use_feedback:
            # 将 classifier 的两个输出拼接在一起
            # char_out: (B, T, num_classes)
            # pos_out: (B, T, num_pos)
            # prev_feedback: (B, T, num_classes + num_pos)
            # 使用 .detach() 断开梯度连接，避免计算图问题
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


# ==================== 训练函数 ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                  use_acceleration=False, weight_decay=1e-4, dropout_rate=0.5, 
                  early_stopping_patience=15, min_delta=0.001):
    """
    训练模型，支持 sector 模式和坐标模式
    
    Args:
        mdl: 模型
        train_data: 训练数据集（MC_RNN_Dataset）
        val_data: 验证数据集（MC_RNN_Dataset）
        num_epochs: 训练轮数
        loss_weights: [字符损失权重, 位置损失权重]，如果为 None 则根据 use_sector 自动设置
                     - sector 模式默认: [1, 1]
                     - 坐标模式默认: [1, 0.001]
        lr: 学习率
        use_acceleration: 是否使用加速训练功能（默认 False）
                         - True: 启用混合精度训练、自动 batch_size 优化、DataLoader 优化等
                         - False: 使用原始训练方式，不影响现有逻辑
        weight_decay: L2 正则化系数（权重衰减），默认 1e-4，用于防止过拟合
        dropout_rate: Dropout 比率，默认 0.5（注意：需要手动修改模型中的 dropout 值）
        early_stopping_patience: 早停耐心值，验证集性能不提升的 epoch 数，默认 15
        min_delta: 早停的最小改进阈值，默认 0.001
    """
    # 从数据集获取 use_sector 信息
    use_sector = train_data.use_sector
    
    # 根据 use_sector 设置默认 loss_weights
    if loss_weights is None:
        if use_sector:
            loss_weights = [1, 1]  # sector 模式：字符和 sector 损失权重相等
        else:
            loss_weights = [1, 0.001]  # 坐标模式：位置损失权重较小（MSE 通常数值较大）
    
    # 按模型内部的 device 来放置参数（可以是 'cuda' 或 'cpu'）
    device = mdl.device
    mdl.to(device)
    
    # ========== 加速训练模块初始化（仅在 use_acceleration=True 时使用）==========
    autocast_fn = None
    GradScaler_cls = None
    scaler = None
    psutil_module = None
    batch_size = 32  # 默认 batch_size
    num_workers = 0   # 默认单进程
    pin_memory = False  # 默认不使用 pin_memory
    show_gpu_usage = False  # 默认不显示 GPU 使用率
    
    if use_acceleration:
        print("启用加速训练功能...")
        autocast_fn, GradScaler_cls, psutil_module = _init_acceleration_modules()
        
        if autocast_fn is None or GradScaler_cls is None:
            print("警告：无法导入加速训练模块，将使用标准训练方式")
            use_acceleration = False
        else:
            # 自动寻找最优 batch_size
            # 注意：GaWFRNNConv 模型跳过 batch_size 搜索，直接使用默认值 32
            # 因为 GaWFRNNConv 使用 feedback 机制，batch_size 变化会导致 prev_feedback 维度不匹配
            if device == 'cuda' and not isinstance(mdl, GaWFRNNConv):
                print("正在自动寻找最优 batch_size...")
                batch_size = _find_optimal_batch_size(mdl, train_data, device=device)
                print(f"使用 batch_size = {batch_size}")
            elif isinstance(mdl, GaWFRNNConv):
                print(f"检测到 GaWFRNNConv 模型，跳过 batch_size 搜索，使用默认 batch_size = {batch_size}")
            
            # 自动设置 num_workers
            if psutil_module is not None:
                num_workers = min(4, psutil_module.cpu_count(logical=False))
            else:
                import os
                num_workers = min(4, os.cpu_count() or 1)
            
            # 启用 pin_memory（仅 GPU）
            pin_memory = (device == 'cuda')
            show_gpu_usage = True
            
            # 初始化混合精度训练的 scaler
            if device == 'cuda':
                scaler = GradScaler_cls('cuda')
                print(f"加速设置: batch_size={batch_size}, num_workers={num_workers}, "
                      f"pin_memory={pin_memory}, 混合精度训练=启用")
            
            # 在加速模式下，限制 batch_size 不超过 32，以保持与原始模式相同的收敛特性
            # 更大的 batch_size 会改变梯度估计的特性，可能导致收敛速度变慢
            # 加速主要通过混合精度训练实现，而不是增大 batch_size
            if batch_size > 32:
                original_batch_size = batch_size
                batch_size = 32
                print(f"batch_size 从 {original_batch_size} 限制为 {batch_size}，以保持与原始模式相同的收敛速度")
    # ========== 加速模块初始化结束 ==========
    
    # 添加权重衰减（L2正则化）来防止过拟合
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_char = nn.CrossEntropyLoss()
    
    # 学习率硬衰减设置：在特定 epoch 处衰减学习率
    # 衰减点：在总 epoch 的 25%, 50%, 75% 处衰减为原来的 0.5 倍
    lr_decay_epochs = [int(num_epochs * 0.25), int(num_epochs * 0.5), int(num_epochs * 0.75)]
    lr_decay_factor = 0.5  # 每次衰减为原来的 0.5 倍
    
    # 学习率预热（warmup）设置：仅在加速模式下使用
    # 在训练初期逐步增加学习率，有助于混合精度训练的稳定性
    use_warmup = use_acceleration and device == 'cuda'
    warmup_epochs = 5 if use_warmup else 0  # 前5个epoch进行预热
    initial_lr = lr  # 保存初始学习率
    
    if use_warmup:
        print(f"学习率预热设置: 前 {warmup_epochs} 个 epoch 从 {lr * 0.1:.6f} 线性增加到 {lr:.6f}")
    print(f"学习率衰减设置: 在 epoch {lr_decay_epochs} 处将学习率衰减为原来的 {lr_decay_factor} 倍")
    
    # 根据 use_sector 选择位置损失函数
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss()  # sector 分类
    else:
        criterion_pos = nn.MSELoss()  # 坐标回归
    
    def loss_fn(out_char, out_pos, labels):
        # 字符损失（两种模式相同）
        labels_char = labels[:, :, 0].long().view(-1)
        outputs_char = out_char.view(-1, out_char.shape[-1])  # (B*T, num_classes)
        loss_char = criterion_char(outputs_char, labels_char)
        
        # 位置损失（根据 use_sector 选择不同方式）
        if use_sector:
            # sector 模式：分类损失
            labels_pos = labels[:, :, 1].long().view(-1)
            outputs_pos = out_pos.view(-1, out_pos.shape[-1])  # (B*T, num_sectors)
            loss_pos = criterion_pos(outputs_pos, labels_pos)
        else:
            # 坐标模式：回归损失（MSE）
            labels_pos = labels[:, :, 1:].float()  # (B, T, 2) -> [x, y]
            outputs_pos = out_pos  # (B, T, 2) -> [x, y]
            loss_pos = criterion_pos(outputs_pos, labels_pos)

        # 与原来的正则保持一致（如果模型中没有 mdl.rnn，需要相应修改）
        rnn_hh = mdl.rnn.weight_hh_l0
        rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
        loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_metric_pos = 0  # sector 模式：准确率；坐标模式：MSE
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                
                # 每个 batch 开始时重置 feedback（如果是 GaWFRNNConv）
                # 这确保 feedback 的 batch_size 和 seq_len 与当前 batch 匹配
                if hasattr(mdl, 'prev_feedback'):
                    mdl.prev_feedback = None
                
                # 根据加速模式选择数据传输方式
                if use_acceleration and pin_memory:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    labels = labels.to(device)
                
                # 根据加速模式选择是否使用混合精度
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
        
        # 返回结果
        acc_char = total_acc_char * 100 / len(data_loader)
        if use_sector:
            metric_pos = total_metric_pos * 100 / len(data_loader)  # 准确率（百分比）
        else:
            metric_pos = total_metric_pos / len(data_loader)  # MSE（像素平方）
        return acc_char, metric_pos

    # data loader（根据加速模式选择不同配置）
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
            shuffle=False,  # 验证集不需要 shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        # 原始方式：简单配置
        train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=32, shuffle=True)
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)  # sector 模式：准确率；坐标模式：MSE
    val_metric_pos = np.zeros(num_epochs)
    
    # 早停机制
    best_val_metric = -np.inf if use_sector else np.inf  # sector模式：越大越好；坐标模式：越小越好
    best_val_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # 学习率预热：在训练初期逐步增加学习率（仅在加速模式下）
        if use_warmup and epoch < warmup_epochs:
            # 线性预热：从 initial_lr * 0.1 线性增加到 initial_lr
            warmup_lr = initial_lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
            for param_group in optim.param_groups:
                param_group['lr'] = warmup_lr
            if epoch == 0 or epoch == warmup_epochs - 1:
                print(f"Epoch {epoch + 1}: 学习率预热到 {warmup_lr:.6f}")
        # 学习率硬衰减：在指定 epoch 处衰减学习率
        elif epoch in lr_decay_epochs:
            current_lr = optim.param_groups[0]['lr']
            new_lr = current_lr * lr_decay_factor
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr
            print(f"Epoch {epoch + 1}: 学习率从 {current_lr:.6f} 衰减到 {new_lr:.6f}")
        
        mdl.train()
        
        # 训练循环
        epoch_train_acc_char = 0.0
        epoch_train_metric_pos = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dl):
            inputs, labels = batch
            
            # 每个 batch 开始时重置 feedback（如果是 GaWFRNNConv）
            # 这确保 feedback 的 batch_size 和 seq_len 与当前 batch 匹配
            if hasattr(mdl, 'prev_feedback'):
                mdl.prev_feedback = None
            
            # 根据加速模式选择数据传输方式
            if use_acceleration and pin_memory:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                labels = labels.to(device)
            
            optim.zero_grad()
            
            # 根据加速模式选择是否使用混合精度训练
            if use_acceleration and scaler is not None and autocast_fn is not None:
                with autocast_fn('cuda'):
                    out_char, out_pos = mdl(inputs)
                    loss = loss_fn(out_char, out_pos, labels)
                
                scaler.scale(loss).backward()
                # 梯度裁剪（混合精度训练）
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                scaler.step(optim)
                scaler.update()
            else:
                # 原始方式：标准训练
                out_char, out_pos = mdl(inputs)
                loss = loss_fn(out_char, out_pos, labels)
                loss.backward()
                # 梯度裁剪（标准训练）
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
            
            # 加速模式：定期清理 GPU 缓存
            if use_acceleration and batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
        
        # 计算 epoch 平均指标
        train_acc_char[epoch] = (epoch_train_acc_char / num_batches) * 100
        if use_sector:
            train_metric_pos[epoch] = (epoch_train_metric_pos / num_batches) * 100  # 准确率（百分比）
        else:
            train_metric_pos[epoch] = epoch_train_metric_pos / num_batches  # MSE（像素平方）
        
        # 格式化输出字符串
        gpu_info = ""
        if use_acceleration and show_gpu_usage and device == 'cuda' and torch.cuda.is_available():
            gpu_mem = _get_gpu_memory_usage()
            gpu_info = f" | GPU 内存: {gpu_mem:.1f}%"
        
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
            
            # 早停检查：根据主要任务（字符识别）的验证准确率来判断
            current_val_metric = val_acc_char[epoch]
            if use_sector:
                # sector模式：使用字符准确率和sector准确率的加权平均
                current_val_metric = (val_acc_char[epoch] + val_metric_pos[epoch]) / 2.0
            
            improved = False
            if use_sector:
                # sector模式：准确率越高越好
                if current_val_metric > best_val_metric + min_delta:
                    improved = True
            else:
                # 坐标模式：MSE越小越好（但这里用字符准确率作为主要指标）
                if current_val_metric > best_val_metric + min_delta:
                    improved = True
            
            if improved:
                best_val_metric = current_val_metric
                best_val_epoch = epoch
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = mdl.state_dict().copy()
                print(f"  ✓ 验证集性能提升！当前最佳: {best_val_metric:.2f} (epoch {epoch + 1})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停触发：验证集性能在 {early_stopping_patience} 个 epoch 内未提升")
                    print(f"最佳验证性能: {best_val_metric:.2f} (epoch {best_val_epoch + 1})")
                    # 恢复最佳模型
                    if best_model_state is not None:
                        mdl.load_state_dict(best_model_state)
                        print("已恢复最佳模型状态")
                    break

    torch.cuda.empty_cache()

    # 如果早停触发，只返回实际训练的epoch数（epoch从0开始，所以实际训练了epoch+1个epoch）
    actual_epochs = epoch + 1
    
    # 根据 use_sector 返回不同的键名，只返回实际训练的epoch数
    if use_sector:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_acc_pos": train_metric_pos[:actual_epochs],  # sector 准确率
            "val_acc_pos": val_metric_pos[:actual_epochs],      # sector 准确率
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # 保存实际训练的epoch数
        }
    else:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_err_pos": train_metric_pos[:actual_epochs],  # 坐标 MSE
            "val_err_pos": val_metric_pos[:actual_epochs],      # 坐标 MSE
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # 保存实际训练的epoch数
        }


# ==================== 工具函数 ====================
def save_results(results, filepath):
    """
    保存训练结果到本地文件
    
    Args:
        results: 训练结果字典
        filepath: 保存路径（例如：'results_rnn' 或 'results_rnn_sector'）
    """
    # 创建保存字典（不包含 model，因为 model 太大）
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # 保存模型状态字典（如果需要的话，可以单独保存）
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # 验证保存前模型的 num_pos
        saved_num_pos = results["model"].fcpos.out_features
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # 保存其他结果
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


# ==================== 主训练代码 ====================
if __name__ == "__main__":
    # 数据路径配置
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
    
    # 创建数据集（可以选择 sector 模式或坐标模式）
    print("Creating datasets...")
    use_sector_mode = True   # 设置为 True 使用 sector 模式，False 使用坐标模式
    use_acceleration = True  # 设置为 True 启用加速训练功能，False 使用原始方式
    
    # 选择模型类型：'rnn'（RNNConv）、'lstm'（LSTMConv）、'gru'（GRUConv）、'gawf'（GaWFRNNConv）
    model_type = "gawf"
    
    if use_sector_mode:
        # sector 模式：3x3 grid -> 9 个 sector
        num_pos = 9  # sector 数量
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True, num_sectors=num_pos)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True, num_sectors=num_pos)
        print("使用 sector 模式（3x3 grid，9 个 sector）")
    else:
        # 坐标模式：直接预测 (x, y) 坐标
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=False)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=False)
        num_pos = 2  # x, y 坐标
        print("使用坐标模式（直接预测 x, y 坐标）")
    
    # 创建模型（位置输出 num_pos 根据模式设置）
    print("Creating model...")
    # 模型类映射表
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

