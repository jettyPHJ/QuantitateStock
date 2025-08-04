import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os


#显示训练参数
def print_model_parameters(model: nn.Module):
    print("模型结构及参数：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50s} shape: {str(list(param.shape)):>20}  参数量: {param.numel():,}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 绘制训练曲线
def plot_train_val_loss(train_losses, val_losses, save_path='loss.png'):
    plt.figure(figsize=(10, 6))

    # === 同步两个 loss 列表长度 ===
    if len(train_losses) < len(val_losses):
        train_losses = [np.nan] + train_losses  # 补一个 NaN 开头

    epochs = range(len(val_losses))

    # 绘图
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
