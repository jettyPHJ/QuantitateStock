import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from model import MambaModel, LSTMAttentionModel
import utils.plot as pl
from data_source.finance.script.block import Block
from data_process.data_set import FinancialDataset, collate_fn

batch_size = 64

# 设置随机种子
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)


def train_model(model: nn.Module, database: FinancialDataset, finetune_flag: bool = False, device=None):
    """训练模型，如果是微调则减小学习率"""

    # 配置数据存储路径
    save_dir = f"model/training_artifacts/{model.__class__.__name__}"
    os.makedirs(save_dir, exist_ok=True)
    file_name, loss_name = 'model.pth', 'loss.png'

    # 生成训练集和测试集
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

    if finetune_flag:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate / 10)
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


def run_experiment(model_cls, pretrain_blocks, finetune_blocks, exclude_stocks=None, use_news=False, mode="both"):
    """
    执行训练流程，可选模式：预训练 / 微调 / 全流程

    :param model_cls: 模型类，例如 MambaStock.MambaModel
    :param pretrain_blocks: 用于预训练的股票板块
    :param finetune_blocks: 用于微调的股票板块
    :param exclude_stocks: 所有阶段统一排除的股票列表
    :param use_news: 是否使用新闻数据
    :param mode: "pretrain" | "finetune" | "both"
    """
    if exclude_stocks is None:
        exclude_stocks = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CUDA版本: {torch.version.cuda}")
    print(f"Using device: {device}")

    model_name = model_cls.__name__
    model_path = f"model/training_artifacts/{model_name}/model.pth"

    # ========== 1. 预训练阶段 ==========
    if mode in ("pretrain", "both"):
        print("==> 开始预训练")

        pretrain_dataset = FinancialDataset(block_codes=pretrain_blocks, use_news=use_news,
                                            exclude_stocks=exclude_stocks)

        model = model_cls(input_dim=len(pretrain_dataset.feature_columns)).to(device)

        train_model(model, pretrain_dataset, device=device)

    # ========== 2. 微调阶段 ==========
    if mode in ("finetune", "both"):
        print("==> 开始微调")

        finetune_dataset = FinancialDataset(block_codes=finetune_blocks, use_news=use_news,
                                            exclude_stocks=exclude_stocks)

        finetune_model = model_cls(input_dim=len(finetune_dataset.feature_columns)).to(device)

        if os.path.exists(model_path):
            finetune_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"成功加载预训练权重: {model_path}")
        else:
            raise FileNotFoundError(f"找不到预训练模型权重: {model_path}")

        train_model(finetune_model, finetune_dataset, finetune_flag=True, device=device)


# --------------------- 使用入口 ---------------------
if __name__ == "__main__":
    run_experiment(
        model_cls=LSTMAttentionModel,
        pretrain_blocks=[Block.get("纳斯达克计算机指数")],
        finetune_blocks=[Block.get("US_芯片")],
        exclude_stocks=["AMD.O"],
        mode="pretrain"  # 可选: "pretrain", "finetune", "both"
    )
