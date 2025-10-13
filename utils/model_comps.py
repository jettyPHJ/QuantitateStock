import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedInputEmbedding(nn.Module):

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mask_attention = nn.Sequential(nn.Linear(input_dim, d_model), nn.Sigmoid())

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)，可能含 NaN
        return: 加权嵌入 (batch_size, seq_len, d_model)
        """
        nan_mask = (~torch.isnan(x)).float()  # (B, T, F)
        x_filled = torch.nan_to_num(x, nan=0.0)  # 替换 nan 为 0
        x_embed = self.embedding(x_filled)  # (B, T, D)
        attn = self.mask_attention(nan_mask)  # (B, T, D)
        return x_embed * attn


class PoolingLayer(nn.Module):

    def __init__(self, method="mean", d_model=None):
        super().__init__()
        self.method = method
        if method == "attention":
            assert d_model is not None
            self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.method == "mean":
            return x.mean(dim=1)
        elif self.method == "last":
            return x[:, -1, :]
        elif self.method == "attention":
            attn_weights = self.attn(x).softmax(dim=1)  # (B, T, 1)
            return (x * attn_weights).sum(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")


class RegressionOutputLayer(nn.Module):
    """
    基于高斯参数建模的回归输出层。
    用于预测下一季度股价变化的概率分布。
    输出分布平滑、可解释：
    - μ 表示期望涨跌幅中心；
    - σ 表示不确定性（波动区间宽度）。
    """

    def __init__(self, d_model: int, dropout_rate: float = 0.3):
        """
        参数:
            d_model (int): 输入特征维度。
            dropout_rate (float): Dropout比例。
        """
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # 输出均值 μ（范围 [-1,1]，即 -100% ~ +100%）
        self.mu_layer = nn.Linear(d_model // 2, 1)
        # 输出标准差 σ（正数，用 softplus 保证）
        self.sigma_layer = nn.Linear(d_model // 2, 1)

        # 定义 21 个离散区间点，用于计算概率分布
        self.register_buffer("weights", torch.linspace(-1.0, 1.0, 21))

    def forward(self, x: torch.Tensor):
        """
        参数:
            x (torch.Tensor): 输入张量 (B, D)
        返回:
            pred (torch.Tensor): 预测期望值 (B, 1)
            probs (torch.Tensor): 对应21个区间的概率分布 (B, 21)
            mu (torch.Tensor): 高斯分布均值 (B, 1)
            sigma (torch.Tensor): 高斯分布标准差 (B, 1)
        """
        h = self.hidden(x)
        mu = torch.tanh(self.mu_layer(h))  # 限制在 [-1,1]
        sigma = F.softplus(self.sigma_layer(h)) + 1e-4  # σ > 0，避免除零

        # 计算每个区间点的高斯概率
        diff = self.weights[None, :] - mu  # (B, 21)
        probs = torch.exp(-0.5 * (diff / sigma)**2)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 归一化为概率

        # 加权求和得到期望预测值
        pred = (probs * self.weights).sum(dim=-1, keepdim=True)
        return pred, probs, mu, sigma
