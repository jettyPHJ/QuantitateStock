import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import model.MambaStock as MambaStock
from data_process.finance_data.database import BlockCode
from data_process.data_set import FinancialDataset, collate_fn

batch_size = 32

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_model(model: nn.Module, database: FinancialDataset):
    """训练模型"""

    train_set, val_set = database.build_datasets()

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 打印模型参数信息
    print(f"可训练参数总量: {count_parameters(model):,}")
    # print_model_parameters(model)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    criterion = nn.MSELoss()

    # 训练循环
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 模型存储路径
    save_dir = f'./model/{model.__class__.__name__}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for origins, batch_features, batch_targets in train_loader:

            origins = [o.to(device) for o in origins]  # origin 是 list[Tensor]，需要单独处理
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            outputs = model(origins, batch_features).squeeze(-1)  # shape: (batch_size,)

            loss = criterion(outputs * 100, batch_targets * 100)

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
            for origins, batch_features, batch_targets in val_loader:

                origins = [o.to(device) for o in origins]
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(origins, batch_features).squeeze(-1)

                loss = criterion(outputs, batch_targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{save_dir}/best_mamba_model.pth')
        else:
            patience_counter += 1

        print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}')
        plot_train_val_loss(train_losses, val_losses, save_path='logs/loss.png')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # 加载训练中验证集表现最好的模型参数
    model.load_state_dict(torch.load(f'{save_dir}/best_mamba_model.pth', weights_only=True))
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


# 演示使用
if __name__ == "__main__":

    print(f"CUDA版本: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可以选择是否使用卷积
    USE_CONV = False  # 设置为False可以禁用卷积
    print(f"开始训练 (使用卷积: {USE_CONV})...")

    # 创建模型
    db = FinancialDataset(block_code=BlockCode.US_CHIP, use_news=False, exclude_stocks=["NVDA.O"])
    model = MambaStock.MambaModel(input_dim=len(db.feature_columns), use_conv=USE_CONV)
    model = model.to(device)

    print(f"Using device: {device}")

    # 训练模型
    train_model(model, db)
