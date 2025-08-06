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


# def manual_financial_model(oringins, params):
#     """
#     使用自定义的金融公式计算每个序列样本的预测股价。
#     - oringins: shape (batch_size, seq_len, input_dim)
#     - params: shape (batch_size, n_params)
#     返回: shape (batch_size,)
#     """
#     # 拆解参数
#     a, b, c = params[:, 0], params[:, 1], params[:, 2]  # shape: (batch_size,)

#     # 取每个序列的最后两个时间步
#     last_1 = torch.stack([x[-1] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
#     last_2 = torch.stack([x[-2] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
#     price = last_1[:, data_set.get_index('平均股价')]
#     ry_now, ry_before = last_1[:, data_set.get_index('TTM营业额')], last_2[:, data_set.get_index('TTM营业额')]
#     gy_now, gy_before = last_1[:, data_set.get_index('TTM毛利率')], last_2[:, data_set.get_index('TTM毛利率')]
#     ny_now, ny_before = last_1[:, data_set.get_index('TTM净利率')], last_2[:, data_set.get_index('TTM净利率')]

#     # 构造手工股价预测公式
#     output = (price * a * ry_now / ry_before * 2 *
#               (b * torch.sigmoid(gy_now - gy_before) + c * torch.sigmoid(ny_now - ny_before)))
#     return output  # (batch_size,)
