import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_comps import MaskedInputEmbedding, PoolingLayer


class MambaModel(nn.Module):
    """Mamba模型"""

    def __init__(self, input_dim=8, d_model=64, n_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.pooling = PoolingLayer(method="last", d_model=d_model)

        # 支持 NaN mask 的输入嵌入层
        self.input_embedding = MaskedInputEmbedding(input_dim, d_model)

        # Mamba层
        self.mamba_layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])

        # 层归一化
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # 确保输出在-1~1之间
        )

    def forward(self, origins, features):
        """
        origins: 原始特征序列 (list of tensors)，可用于构造手工 financial model 计算。
        features: 归一化后的输入特征,shape (batch_size, seq_len, input_dim)
        """
        # 输入嵌入 支持 NaN 注意力加权
        x = self.input_embedding(features)  # (batch_size, seq_len, d_model)

        # Mamba 层 + 残差连接
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # 池化
        x_pooled = self.pooling(x)  # (batch_size, d_model)

        # 输出参数预测： 预测涨幅范围
        output = self.output_layer(x_pooled)
        return output


class MambaBlock(nn.Module):
    """核心Mamba块实现"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # 激活函数
        self.activation = nn.SiLU()

        # 初始化SSM参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        # 输入投影
        xz = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_inner)

        x_ac = self.activation(x_proj)

        # SSM
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 计算 B, C, dt
        x_dbl = self.x_proj(x_ac)  # (batch_size, seq_len, d_state * 2)
        B, C = x_dbl.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_state)

        dt = self.dt_proj(x_ac)  # (batch_size, seq_len, d_inner)
        dt = F.softplus(dt)

        # 简化的SSM计算
        y = self.selective_scan(x_ac, dt, A, B, C, self.D)

        # 门控
        y = y * self.activation(z)

        # 输出投影
        output = self.out_proj(y)

        return output

    def selective_scan(self, u, delta, A, B, C, D):
        # 简化实现：使用RNN风格的递归计算
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # 初始化隐状态
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)

        outputs = []
        for i in range(seq_len):
            # 当前时间步的输入
            u_i = u[:, i, :]  # (batch_size, d_inner)
            delta_i = delta[:, i, :].unsqueeze(-1)  # (batch_size, d_inner, 1)
            B_i = B[:, i, :]  # (batch_size, d_state)
            C_i = C[:, i, :]  # (batch_size, d_state)

            # 状态更新
            # dA = exp(delta_i * A)
            dA = torch.exp(delta_i * A)  # (batch_size, d_inner, d_state)
            # dB = delta_i * B_i
            dB = delta_i * B_i.unsqueeze(1)  # (batch_size, d_inner, d_state)

            h = h * dA + dB * u_i.unsqueeze(-1)  # (batch_size, d_inner, d_state)

            # 输出计算
            # y_i = (C_i @ h) + D * u_i
            # 这里的 C_i 需要转置并与 h 的最后两维进行点积
            y_i = torch.sum(h * C_i.unsqueeze(1), dim=-1) + D * u_i  # (batch_size, d_inner)
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_inner)
