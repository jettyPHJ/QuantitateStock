import torch
import torch.nn as nn
from utils.model_comps import MaskedInputEmbedding

import torch
import torch.nn as nn


class MultiHeadTemporalPooling(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model > 0 and num_heads > 0
        self.num_heads = num_heads
        self.attn_proj = nn.Linear(d_model, num_heads)
        # 学习如何融合多个头的信息
        self.head_fusion = nn.Linear(num_heads, 1)

    def forward(self, x):
        """
        x: (B, T, D) - 输入序列
        return: (B, D) - 加权池化结果
        """
        B, T, D = x.shape
        attn_logits = self.attn_proj(x)  # (B, T, H)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, H)

        # 扩展维度以做广播相乘
        weighted = x.unsqueeze(-1) * attn_weights.unsqueeze(-2)  # (B, T, D, H)
        pooled_heads = weighted.sum(dim=1)  # (B, D, H)

        # 修正：直接在 H 维度上应用融合层，无需转置
        fused = self.head_fusion(pooled_heads).squeeze(-1)  # (B, D)
        return fused


class LSTMAttentionModel(nn.Module):

    def __init__(self, input_dim=8, d_model=64, num_layers=1, bidirectional=False, num_attn_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.hidden_dim = d_model if not bidirectional else d_model // 2

        self.input_embedding = MaskedInputEmbedding(input_dim, d_model)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)

        self.pooling = MultiHeadTemporalPooling(d_model=d_model, num_heads=num_attn_heads)

        self.output_layer = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
                                          nn.Linear(d_model // 2, 1), nn.Tanh())

    def forward(self, origins, features):
        """
        origins: list of raw tensors (unused in this model, kept for compatibility)
        features: (B, T, F)
        """
        x_embed = self.input_embedding(features)  # (B, T, D)
        lstm_out, _ = self.lstm(x_embed)  # (B, T, D)
        pooled = self.pooling(lstm_out)  # (B, D)
        output = self.output_layer(pooled)  # (B, 1)
        return output
