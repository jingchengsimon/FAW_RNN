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


# ==================== 数据集类 ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=10):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): 如果为 True，则把 (x, y) 位置映射为 0-(num_sectors-1) 的 sector id
            num_sectors (int): sector 的数量，例如 10 表示 0-9 共 10 个 sector
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
            # 使用图像宽度把 x 坐标均分到 num_sectors 个区间，得到 0-(num_sectors-1) 的 sector id
            width = self.data.shape[-1]
            x = labels[:, 1].astype(np.float32)
            sector = (x / max(width - 1, 1) * self.num_sectors).astype(np.int64)
            sector = np.clip(sector, 0, self.num_sectors - 1)
            # 新的 label: [char_id, sector_id]
            labels = np.stack([labels[:, 0].astype(np.int64), sector], axis=1)

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


# ==================== 训练函数 ====================
def network_train_sector(mdl, train_data, val_data, num_epochs=50, loss_weights=[1, 1], lr=0.001):
    """
    训练 sector 模式的模型
    
    Args:
        mdl: 模型
        train_data: 训练数据集
        val_data: 验证数据集
        num_epochs: 训练轮数
        loss_weights: [字符损失权重, 位置损失权重]
        lr: 学习率
    """
    # 按模型内部的 device 来放置参数（可以是 'cuda' 或 'cpu'）
    mdl.to(mdl.device)
    optim = torch.optim.Adam(mdl.parameters(), lr=lr)
    criterion_char = nn.CrossEntropyLoss()
    criterion_pos = nn.CrossEntropyLoss()  # sector 分类

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
                labels = labels.to(mdl.device)
                out_char, out_pos = mdl(inputs)
                # char 精度
                total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                # sector 精度
                total_acc_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
        return total_acc_char * 100 / len(data_loader), total_acc_pos * 100 / len(data_loader)

    # data loader
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=32, shuffle=True)
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_acc_pos = np.zeros(num_epochs)
    val_acc_pos = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        mdl.train()
        for batch in train_dl:
            inputs, labels = batch
            labels = labels.to(mdl.device)
            optim.zero_grad()
            out_char, out_pos = mdl(inputs)
            loss = loss_fn(out_char, out_pos, labels)
            loss.backward()
            optim.step()

            train_acc_char[epoch] += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            train_acc_pos[epoch] += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()

        train_acc_char[epoch] /= len(train_dl)
        train_acc_char[epoch] *= 100
        train_acc_pos[epoch] /= len(train_dl)
        train_acc_pos[epoch] *= 100
        train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_acc_pos[epoch]:.2f}%)"

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
def model_param_count(mdl):
    """计算模型参数数量"""
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def save_results(results, filepath):
    """
    保存训练结果到本地文件
    
    Args:
        results: 训练结果字典
        filepath: 保存路径（例如：'results_rnn.pkl' 或 'results_rnn_sector.pkl'）
    """
    # 创建保存字典（不包含 model，因为 model 太大）
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # 保存模型状态字典（如果需要的话，可以单独保存）
    model_path = filepath.replace('.pkl', '_model.pth')
    if "model" in results:
        torch.save(results["model"].state_dict(), model_path)
        save_dict["model_path"] = model_path
        print(f"Model state dict saved to: {model_path}")
    
    # 保存其他结果
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {filepath}")


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
    
    # 创建数据集（sector 模式）
    print("Creating datasets...")
    train_ds_sector = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True)
    val_ds_sector = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True)
    
    # 创建模型
    print("Creating model...")
    mdl_rnn = RNNConv(num_classes=10, num_pos=10, kernel_size=5)
    print(f"RNN model parameter count: {model_param_count(mdl_rnn)}")
    
    # 训练模型
    print("Starting training...")
    results_rnn_sector = network_train_sector(
        mdl_rnn, 
        train_ds_sector, 
        val_ds_sector, 
        num_epochs=200, 
        loss_weights=[1.0, 1.0]
    )
    
    # 保存训练结果
    print("\nSaving results...")
    save_results(results_rnn_sector, "results_rnn_sector.pkl")
    
    print("\nTraining completed!")

