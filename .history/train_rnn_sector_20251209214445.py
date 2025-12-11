"""
独立运行的 RNN Sector 训练脚本
用于训练 RNN 模型并保存结果
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

try:
    import psutil
except ImportError:
    psutil = None


# ==================== 数据集类 ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): 如果为 True，则把 (x, y) 位置映射为 0-(num_sectors-1) 的 sector id
            num_sectors (int): sector 的数量，例如 9 表示 0-8 共 9 个 sector（3x3 grid）
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
            # 使用图像宽度和高度，把 (x, y) 映射到一个 grid_size x grid_size 的网格上，
            # 得到 0-(num_sectors-1) 的 sector id（例如 num_sectors=9 -> 3x3 grid）
            height = self.data.shape[-2]
            width = self.data.shape[-1]

            # 根据 num_sectors 推出每一维的 grid_size（假设 num_sectors 是完全平方数，如 9, 16 等）
            grid_size = int(np.sqrt(self.num_sectors))
            if grid_size * grid_size != self.num_sectors:
                raise ValueError(f"num_sectors={self.num_sectors} 不是完全平方数，无法构成 grid_size x grid_size 网格")

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

        # 转换为 torch tensor 以避免 pin_memory 问题
        stacked_frames = torch.from_numpy(stacked_frames).contiguous()
        labels = torch.from_numpy(labels).contiguous()
        
        return stacked_frames, labels


# ==================== 模型类 ====================
class RNNConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda'):
        super(RNNConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        # 原始 RNN 版本
        self.rnn = nn.RNN(input_size=64 * 12 * 12, hidden_size=256,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(256)
        self.fcchar = nn.Linear(256, num_classes)
        self.fcpos = nn.Linear(256, num_pos)
        self.to(self.device)

    def encoder(self, x):
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
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


class GRUConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda'):
        super(GRUConv, self).__init__()
        self.device = device
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
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
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
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda'):
        super(LSTMConv, self).__init__()
        self.device = device
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
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
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


# ==================== GPU 工具函数 ====================
def get_gpu_memory_usage():
    """获取当前 GPU 内存使用率（百分比）"""
    if not torch.cuda.is_available():
        return 0.0
    # 使用 reserved 内存来计算使用率更准确
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    return (reserved / total) * 100.0 if total > 0 else 0.0

def find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256):
    """
    自动寻找最优 batch_size，在 GPU 不超载的情况下
    
    Args:
        model: 模型
        train_data: 训练数据集
        device: 设备
        start_batch_size: 起始 batch_size
        max_batch_size: 最大 batch_size
    
    Returns:
        最优 batch_size
    """
    if device == 'cpu':
        return start_batch_size
    
    model.eval()
    optimal_batch_size = start_batch_size
    
    # 测试不同的 batch_size
    for batch_size in [start_batch_size, 64, 128, 256]:
        if batch_size > max_batch_size:
            break
        
        try:
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 创建测试数据
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_batch = next(iter(test_loader))
            inputs, labels = test_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播测试
            with torch.no_grad():
                _ = model(inputs)
            
            # 检查内存使用
            memory_usage = get_gpu_memory_usage()
            
            if memory_usage < 80.0:  # 如果内存使用率小于 80%，可以尝试更大的 batch_size
                optimal_batch_size = batch_size
                print(f"测试 batch_size={batch_size}: GPU 内存使用率 {memory_usage:.1f}%, 可以使用")
            else:
                print(f"测试 batch_size={batch_size}: GPU 内存使用率 {memory_usage:.1f}%, 超出限制")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} 导致 OOM，使用 batch_size={optimal_batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    torch.cuda.empty_cache()
    model.train()
    return optimal_batch_size


# ==================== 训练函数 ====================
def network_train_sector(mdl, train_data, val_data, num_epochs=50, loss_weights=[1, 1], lr=0.001, 
                         batch_size=None, use_amp=True, num_workers=None, pin_memory=True):
    """
    训练 sector 模式的模型
    
    Args:
        mdl: 模型
        train_data: 训练数据集
        val_data: 验证数据集
        num_epochs: 训练轮数
        loss_weights: [字符损失权重, 位置损失权重]
        lr: 学习率
        batch_size: batch 大小，如果为 None 则自动寻找最优值
        use_amp: 是否使用混合精度训练（FP16）
        num_workers: DataLoader 的 num_workers，如果为 None 则自动设置
        pin_memory: 是否使用 pin_memory 加速数据传输
    """
    # 按模型内部的 device 来放置参数（可以是 'cuda' 或 'cpu'）
    device = mdl.device
    mdl.to(device)
    
    # 自动设置 num_workers（根据 CPU 核心数）
    if num_workers is None:
        if psutil is not None:
            num_workers = min(4, psutil.cpu_count(logical=False))  # 使用物理核心数，最多 4 个
        else:
            num_workers = min(4, os.cpu_count() or 1)  # fallback 到 os.cpu_count()
    
    # 自动寻找最优 batch_size
    if batch_size is None:
        print("正在自动寻找最优 batch_size...")
        batch_size = find_optimal_batch_size(mdl, train_data, device=device)
        print(f"使用 batch_size = {batch_size}")
    
    optim = torch.optim.Adam(mdl.parameters(), lr=lr)
    criterion_char = nn.CrossEntropyLoss()
    criterion_pos = nn.CrossEntropyLoss()  # sector 分类
    
    # 混合精度训练
    if use_amp and device == 'cuda':
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            # 兼容旧版本
            scaler = GradScaler()
    else:
        scaler = None

    def loss_fn(out_char, out_pos, labels):
        # labels: (B, T, 2) -> [char_id, sector_id]
        labels_char = labels[:, :, 0].long().view(-1)
        labels_pos = labels[:, :, 1].long().view(-1)
        outputs_char = out_char.view(-1, out_char.shape[-1])      # (B*T, num_classes)
        outputs_pos = out_pos.view(-1, out_pos.shape[-1])         # (B*T, num_sectors)
        loss_char = criterion_char(outputs_char, labels_char)
        loss_pos = criterion_pos(outputs_pos, labels_pos)

        # 与原来的正则保持一致（如果模型中没有 mdl.rnn，需要相应修改）
        rnn_hh = mdl.rnn.weight_hh_l0
        rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
        loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_acc_pos = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                
                # 验证时也使用混合精度以加快速度
                if use_amp and scaler is not None:
                    try:
                        with autocast(device_type=device if device != 'cpu' else 'cpu'):
                            out_char, out_pos = mdl(inputs)
                    except TypeError:
                        # 兼容旧版本
                        with autocast():
                            out_char, out_pos = mdl(inputs)
                else:
                    out_char, out_pos = mdl(inputs)
                
                # char 精度
                total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                # sector 精度
                total_acc_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
        return total_acc_char * 100 / len(data_loader), total_acc_pos * 100 / len(data_loader)

    # data loader（优化设置）
    train_dl = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        persistent_workers=num_workers > 0  # 保持 worker 进程，避免重复创建
    )
    val_dl = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,  # 验证集不需要 shuffle
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        persistent_workers=num_workers > 0
    )
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_acc_pos = np.zeros(num_epochs)
    val_acc_pos = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        mdl.train()
        for batch_idx, batch in enumerate(train_dl):
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            
            optim.zero_grad()
            
            # 混合精度训练
            if use_amp and scaler is not None:
                try:
                    with autocast(device_type=device if device != 'cpu' else 'cpu'):
                        out_char, out_pos = mdl(inputs)
                        loss = loss_fn(out_char, out_pos, labels)
                except TypeError:
                    # 兼容旧版本
                    with autocast():
                        out_char, out_pos = mdl(inputs)
                        loss = loss_fn(out_char, out_pos, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out_char, out_pos = mdl(inputs)
                loss = loss_fn(out_char, out_pos, labels)
                loss.backward()
                optim.step()

            train_acc_char[epoch] += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            train_acc_pos[epoch] += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
            
            # 定期清理缓存（每 100 个 batch）
            if batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

        train_acc_char[epoch] /= len(train_dl)
        train_acc_char[epoch] *= 100
        train_acc_pos[epoch] /= len(train_dl)
        train_acc_pos[epoch] *= 100
        
        # 显示 GPU 使用情况
        gpu_info = ""
        if device == 'cuda' and torch.cuda.is_available():
            gpu_mem = get_gpu_memory_usage()
            gpu_info = f" | GPU 内存: {gpu_mem:.1f}%"
        
        train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_acc_pos[epoch]:.2f}%){gpu_info}"

        with torch.no_grad():
            val_acc_char[epoch], val_acc_pos[epoch] = evaluate(mdl, val_dl)
            val_str = f" Validation (char, sector): ({val_acc_char[epoch]:.2f}%, {val_acc_pos[epoch]:.2f}%)"
            print(train_str, val_str)

    torch.cuda.empty_cache()

    return {
        "train_acc_char": train_acc_char,
        "val_acc_char": val_acc_char,
        "train_acc_pos": train_acc_pos,
        "val_acc_pos": val_acc_pos,
        "model": mdl.to("cpu")
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
    
    # 创建数据集（sector 模式，3x3 grid -> 9 个 sector）
    print("Creating datasets...")
    train_ds_sector = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True, num_sectors=9)
    val_ds_sector = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True, num_sectors=9)
    
    # 创建模型（位置输出 num_pos 与 sector 数量一致，这里为 9）
    print("Creating model...")
    mdl_rnn = RNNConv(num_classes=10, num_pos=9, kernel_size=5)
    
    # 训练模型（使用优化设置）
    print("Starting training...")
    print("优化设置:")
    print("  - 自动寻找最优 batch_size")
    print("  - 使用混合精度训练 (FP16)")
    print("  - 优化 DataLoader (num_workers, pin_memory)")
    results_rnn_sector = network_train_sector(
        mdl_rnn, 
        train_ds_sector, 
        val_ds_sector, 
        num_epochs=200, 
        loss_weights=[1.0, 1.0],
        batch_size=None,  # 自动寻找最优值
        use_amp=True,     # 使用混合精度训练
        num_workers=None, # 自动设置
        pin_memory=True   # 加速数据传输
    )
    
    # 保存训练结果
    print("\nSaving results...")
    save_results(results_rnn_sector, "results_rnn_sector_ori")
    
    print("\nTraining completed!")

