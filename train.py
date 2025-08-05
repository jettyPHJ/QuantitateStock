import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import model.MambaStock as MambaStock
import utils.plot as pl
from data_process.finance_data.database import BlockCode
from data_process.data_set import FinancialDataset, collate_fn

batch_size = 64

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_model(model: nn.Module, database: FinancialDataset, finetune_flag: bool = False):
    """训练模型，如果是微调则减小学习率"""

    train_set, val_set = database.build_datasets()

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 打印模型参数信息
    print(f"是否微调:{finetune_flag} 可训练参数总量: {pl.count_parameters(model):,}")

    # 优化器
    learning_rate, decay_patience = 1e-3, 10
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=decay_patience, factor=0.5)

    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 模型存储路径
    save_dir = f'./model/{model.__class__.__name__}'
    os.makedirs(save_dir, exist_ok=True)
    file_name, loss_name = 'model.pth', 'loss.png'

    if finetune_flag:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate / 10, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=decay_patience / 2, factor=0.5)
        patience = patience / 2
        file_name, loss_name = 'model_finetune.pth', 'loss_finetune.png'

    for epoch in range(100):
        if epoch != 0:
            model.train()
            train_loss = 0
            for origins, batch_features, batch_targets in train_loader:
                origins = [o.to(device) for o in origins]
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(origins, batch_features).squeeze(-1)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
        else:
            train_loss = float('nan')  # 避免打印时报错

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{save_dir}/{file_name}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        pl.plot_train_val_loss(train_losses, val_losses, save_path=f'{save_dir}/{loss_name}')

    model.load_state_dict(torch.load(f'{save_dir}/{file_name}', weights_only=True))
    return model


# 演示使用
if __name__ == "__main__":

    print(f"CUDA版本: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    db = FinancialDataset(block_codes=[BlockCode.NASDAQ_Computer_Index], use_news=False, exclude_stocks=["NVDA.O"])
    model = MambaStock.MambaModel(input_dim=len(db.feature_columns))
    model = model.to(device)

    print(f"Using device: {device}")

    # 训练模型
    train_model(model, db)
