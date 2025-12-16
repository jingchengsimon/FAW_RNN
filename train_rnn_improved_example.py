"""
改进的训练配置示例 - 用于减少过拟合

主要改进：
1. 启用Early Stopping
2. 增加Dropout和Weight Decay
3. 添加学习率调度器
4. 添加Label Smoothing
5. 在分类器层添加Dropout
"""

# ==================== 改进1: 修改network_train函数签名 ====================
# 在network_train函数中添加label_smoothing参数
def network_train_improved(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                  use_acceleration=False, weight_decay=1e-4, dropout_rate=0.5, 
                  use_early_stopping=True, early_stopping_patience=15, min_delta=0.001,
                  label_smoothing=0.1, use_lr_scheduler=True):  # 新增参数
    """
    改进的训练函数
    
    新增参数：
        label_smoothing: Label smoothing系数，默认0.1
        use_lr_scheduler: 是否使用学习率调度器，默认True
    """
    # ... 前面的代码保持不变 ...
    
    # ==================== 改进2: 修改损失函数，添加Label Smoothing ====================
    # 原代码：
    # criterion_char = nn.CrossEntropyLoss()
    # criterion_pos = nn.CrossEntropyLoss()  # sector mode
    
    # 改进后：
    criterion_char = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion_pos = nn.MSELoss()
    
    # ==================== 改进3: 添加学习率调度器 ====================
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 添加学习率调度器
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='max',  # 监控验证准确率（越大越好）
            factor=0.5,  # 学习率衰减因子
            patience=5,  # 5个epoch无改善则降低学习率
            verbose=True,  # 打印学习率变化
            min_lr=1e-6  # 最小学习率
        )
    
    # ==================== 改进4: 在训练循环中使用调度器 ====================
    # 在验证后添加：
    # if scheduler is not None:
    #     # 使用验证准确率作为监控指标
    #     if use_sector:
    #         monitor_metric = (val_acc_char[epoch] + val_metric_pos[epoch]) / 2.0
    #     else:
    #         monitor_metric = val_acc_char[epoch]
    #     scheduler.step(monitor_metric)
    
    # ... 其余代码保持不变 ...


# ==================== 改进5: 修改模型类，在分类器层添加Dropout ====================
class RNNConv_Improved(nn.Module):
    """改进的RNNConv模型，在分类器层添加dropout"""
    
    def classifier(self, x):
        # 添加dropout
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fcchar(x), self.fcpos(x)


# ==================== 改进6: 主训练代码配置示例 ====================
"""
# 在主训练代码中应用改进：

# 方案A：保守改进（推荐先试）
results_rnn = network_train(
    mdl_rnn, 
    train_ds, 
    val_ds, 
    num_epochs=200, 
    use_acceleration=use_acceleration,
    weight_decay=5e-4,  # 从1e-4增加到5e-4
    dropout_rate=0.5,   # 从0.3增加到0.5
    use_early_stopping=True,  # 启用early stopping
    early_stopping_patience=10,  # 10个epoch无改善则停止
    min_delta=0.001,
    label_smoothing=0.1,  # 添加label smoothing
    use_lr_scheduler=True  # 启用学习率调度器
)

# 方案B：激进改进（如果方案A效果不佳）
results_rnn = network_train(
    mdl_rnn, 
    train_ds, 
    val_ds, 
    num_epochs=200, 
    use_acceleration=use_acceleration,
    weight_decay=1e-3,  # 增加到1e-3
    dropout_rate=0.6,   # 增加到0.6
    use_early_stopping=True,
    early_stopping_patience=8,
    min_delta=0.001,
    label_smoothing=0.15,  # 更大的label smoothing
    use_lr_scheduler=True
)
"""


# ==================== 改进7: 修改RNN层的Dropout率 ====================
"""
在模型定义中修改middle方法：

def middle(self, x):
    x = self.rnn(x)[0]
    x = self.LNormRNN(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.6, training=self.training)  # 从0.5增加到0.6或0.7
    return x
"""

