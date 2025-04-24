import torch
import torch.nn as nn



class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        """
        残差连接，用于在每个子层后添加残差连接和 Dropout。

        参数:
            dropout: Dropout 概率，用于在残差连接前应用于子层输出，防止过拟合。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        """
        前向传播函数。

        参数:
            x: 残差连接的输入张量，形状为 (batch_size, seq_len, d_model)。
            sublayer: 子层模块的函数，多头注意力或前馈网络。

        返回:
            经过残差连接和 Dropout 处理后的张量，形状为 (batch_size, seq_len, d_model)。
        """
        return self.dropout(sublayer(x)) + x

# 模拟一个子层，比如前馈网络的一层
class DummySublayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 3
    d_model = 4

    # 构造输入张量
    x = torch.randn(batch_size, seq_len, d_model)

    # 实例化子层和残差连接模块
    dummy_sublayer = DummySublayer(d_model)
    residual = ResidualConnection(dropout=0.1)

    # 使用 ResidualConnection 包裹子层
    output = residual(x, dummy_sublayer)

    print("输入：\n", x)
    print("输出（残差连接后）：\n", output)