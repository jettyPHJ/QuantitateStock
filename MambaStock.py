import torch
import torch.nn as nn
import torch.nn.functional as F
import data_process.data_set as data_set

d_model = 128
n_layers = 5
n_params = 3


class MambaBlock(nn.Module):
    """简化版Mamba块实现"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_conv=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.use_conv = use_conv

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 可选的卷积层
        if self.use_conv:
            self.conv1d = nn.Conv1d(in_channels=self.d_inner,
                                    out_channels=self.d_inner,
                                    kernel_size=d_conv,
                                    padding=d_conv - 1,
                                    groups=self.d_inner)

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
        # x shape: (batch_size, seq_len, d_model)
        _, seq_len, _ = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_inner)

        # 可选的卷积处理
        if self.use_conv:
            x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            x_conv = self.activation(x_conv)
        else:
            x_conv = self.activation(x_proj)

        # SSM
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 计算 B, C, dt
        x_dbl = self.x_proj(x_conv)  # (batch_size, seq_len, d_state * 2)
        B, C = x_dbl.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_state)

        dt = self.dt_proj(x_conv)  # (batch_size, seq_len, d_inner)
        dt = F.softplus(dt)

        # 简化的SSM计算
        y = self.selective_scan(x_conv, dt, A, B, C, self.D)

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
            B_i = B[:, i, :].unsqueeze(1)  # (batch_size, 1, d_state)
            C_i = C[:, i, :].unsqueeze(1)  # (batch_size, 1, d_state)

            # 状态更新
            dA = torch.exp(delta_i * A.unsqueeze(0))  # (batch_size, d_inner, d_state)
            dB = delta_i * B_i  # (batch_size, d_inner, d_state)

            h = h * dA + dB * u_i.unsqueeze(-1)  # (batch_size, d_inner, d_state)

            # 输出计算
            y_i = torch.sum(h * C_i, dim=-1) + D * u_i  # (batch_size, d_inner)
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_inner)


class MambaModel(nn.Module):
    """完整的Mamba模型"""

    def __init__(self,
                 input_dim=len(data_set.feature_columns),
                 d_model=d_model,
                 n_layers=n_layers,
                 n_params=n_params,
                 use_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_params = n_params

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Mamba层
        self.mamba_layers = nn.ModuleList([MambaBlock(d_model, use_conv=use_conv) for _ in range(n_layers)])

        # 层归一化
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_params),
            nn.Sigmoid()  # 确保输出在0-1之间
        )

    def forward(self, origins, x, lengths=None):

        _, seq_len, _ = x.shape

        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Mamba 层 + 残差连接
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # TODO：应该改成可学习的池化 ，最好封装成模块，对比mean pooling，last-step pooling，attention pooling，
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

            x_masked = x * mask
            x_pooled = x_masked.sum(dim=1) / (lengths.unsqueeze(1) + 1e-8)
        else:
            x_pooled = x.mean(dim=1)

        # 输出参数预测
        params = self.output_layer(x_pooled)  # (batch_size, n_params)
        # 确保参数在合理范围内
        params = params * 2  # output ∈ (0, 3)
        # 自定义公式输出
        output = self.manual_financial_model(origins, params)
        return output

    def manual_financial_model(self, oringins, params):
        """
        使用自定义的金融公式计算每个序列样本的预测股价。
        - oringins: shape (batch_size, seq_len, input_dim)
        - params: shape (batch_size, n_params)
        返回: shape (batch_size,)
        """
        # 拆解参数
        a, b, c = params[:, 0], params[:, 1], params[:, 2]  # shape: (batch_size,)

        # 取每个序列的最后两个时间步
        last_1 = torch.stack([x[-1] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
        last_2 = torch.stack([x[-2] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
        price = last_1[:, data_set.get_index('平均股价')]
        ry_now, ry_before = last_1[:, data_set.get_index('TTM营业额')], last_2[:, data_set.get_index('TTM营业额')]
        gy_now, gy_before = last_1[:, data_set.get_index('TTM毛利率')], last_2[:, data_set.get_index('TTM毛利率')]
        ny_now, ny_before = last_1[:, data_set.get_index('TTM净利率')], last_2[:, data_set.get_index('TTM净利率')]

        # 构造手工股价预测公式
        output = (price * a * ry_now / ry_before * 2 *
                  (b * torch.sigmoid(gy_now - gy_before) + c * torch.sigmoid(ny_now - ny_before)))
        return output  # (batch_size,)
