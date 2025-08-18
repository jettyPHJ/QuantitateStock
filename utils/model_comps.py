import torch
import torch.nn as nn


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
