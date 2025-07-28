import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import data_process.data_set as data_set
import os
import MambaStock

batch_size = 64

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class FinancialDataset(Dataset):
    """财务数据集"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        origins = torch.tensor(sample['origin'], dtype=torch.float32)  # shape: (seq_len, input_dim)
        features = torch.tensor(sample['features'], dtype=torch.float32)  # shape: (seq_len, input_dim)
        target = torch.tensor([sample['target']], dtype=torch.float32)  # shape: (1,)
        return origins, features, target


def collate_fn(batch):
    """
    batch: List of tuples from Dataset.__getitem__:
        Each tuple: (origin: Tensor(seq_len, input_dim),
                     features: Tensor(seq_len, input_dim),
                     target: Tensor(()))
    """
    max_len = max(f.shape[0] for _, f, _ in batch)
    input_dim = batch[0][1].shape[1]  # features.shape[1]

    batch_origins = []
    batch_features = []
    batch_targets = []
    lengths = []

    for origins, features, targets in batch:
        lengths.append(features.shape[0])  # 保存真实长度
        pad_len = max_len - features.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, input_dim)
            features = torch.cat([features, pad], dim=0)

        batch_origins.append(origins)
        batch_features.append(features)
        batch_targets.append(targets)

    batch_features = torch.stack(batch_features)  # (batch_size, max_seq_len, input_dim)
    batch_targets = torch.stack(batch_targets).squeeze()  # (batch_size,)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return batch_origins, batch_features, batch_targets, lengths


def AdaptiveMAPE_loss(outputs, targets, min_output=0.05, fallback_weight=0.1):
    """
    自适应 MAPE 损失函数:
    - 以预测值为分母，模拟“决策偏差”
    - 对 outputs 太小时使用 fallback MAE 避免不稳定
    """
    outputs_safe = outputs.clone()

    # 防止除以极小值
    mask = outputs.abs() < min_output
    outputs_safe[mask] = min_output

    # MAPE 成分（投资视角）
    percentage_error = torch.abs((targets - outputs) / (targets))

    # Fallback：当 outputs 太小，偏向用 MAE
    fallback_mae = torch.abs(targets - outputs)

    # 自适应融合
    loss = percentage_error * (~mask) + fallback_weight * fallback_mae * mask.float()

    return torch.mean(loss) * 100


def train_model(model):
    """训练模型"""
    raw_data = data_set.load_excel(data_set.file_path, data_set.company_names, data_set.feature_columns)
    data = data_set.generate_synthetic_data(raw_data)

    # 分割数据集
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # 创建数据加载器
    train_dataset = FinancialDataset(train_data)
    val_dataset = FinancialDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 打印模型参数信息
    print(f"可训练参数总量: {count_parameters(model):,}")
    # print_model_parameters(model)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for origins, batch_features, batch_targets, lengths in train_loader:

            origins = [o.to(device) for o in origins]  # origin 是 list[Tensor]，需要单独处理
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            outputs = model(origins, batch_features, lengths)  # shape: (batch_size,)
            loss = AdaptiveMAPE_loss(outputs, batch_targets)  # batch_targets: (batch_size,)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for origins, batch_features, batch_targets, lengths in val_loader:

                origins = [o.to(device) for o in origins]
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                lengths = lengths.to(device)

                outputs = model(origins, batch_features, lengths)
                loss = AdaptiveMAPE_loss(outputs, batch_targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mamba_model.pth')
        else:
            patience_counter += 1

        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        plot_train_val_loss(train_losses, val_losses, save_path='logs/loss.png')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    # 加载训练中验证集表现最好的模型参数
    model.load_state_dict(torch.load('best_mamba_model.pth', weights_only=True))
    return model


#显示训练参数
def print_model_parameters(model):
    print("模型结构及参数：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50s} shape: {str(list(param.shape)):>20}  参数量: {param.numel():,}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 绘制训练曲线
def plot_train_val_loss(train_losses, val_losses, save_path='loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    # 保存并关闭
    plt.savefig(save_path)
    plt.close()
    return


def predict_new_company(model, scaler, quarters_data):
    """为新公司预测参数"""
    # 标准化输入数据
    normalized_data = scaler.transform(quarters_data)

    # 转换为张量
    features = torch.FloatTensor(normalized_data).unsqueeze(0)  # 添加batch维度
    length = torch.LongTensor([len(quarters_data)])

    # 预测
    model.eval()
    with torch.no_grad():
        params = model(features, length)

    return params.squeeze().numpy()


# 演示使用
if __name__ == "__main__":

    print(f"CUDA版本: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可以选择是否使用卷积
    USE_CONV = False  # 设置为False可以禁用卷积
    print(f"开始训练 (使用卷积: {USE_CONV})...")

    # 创建模型
    feature_dim = len(data_set.feature_columns)  # 财务特征维度
    model = MambaStock.MambaModel(use_conv=USE_CONV)
    model = model.to(device)

    print(f"Using device: {device}")

    # 训练模型
    train_model(model)
