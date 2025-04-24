import torch
import torch.nn as nn
import LayerNorm
import  ResidualConnection

class SublayerConnection(nn.Module):
    def __init__(self,dropout, feature_size, epsilon=1e-9):
        super().__init__()
        self.norm = LayerNorm.LayerNorm(feature_size, epsilon)
        self.residual = ResidualConnection.ResidualConnection(dropout)
    def forward(self, x, sublayer):

        output = self.residual(x, sublayer)
        return self.norm(output)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    feature_size = 8

    # 输入张量
    x = torch.randn(batch_size, seq_len, feature_size)

    # 假装的子层，做个线性变换（比如 Attention 或 FFN）
    fake_sublayer = nn.Linear(feature_size, feature_size)

    # 包装成函数，以符合 sublayer(x) 的形式
    sublayer_fn = lambda x: fake_sublayer(x)

    # 创建 SublayerConnection 实例
    sublayer_conn = SublayerConnection(0.3, feature_size)

    # 执行前向传播
    output = sublayer_conn(x, sublayer_fn)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("输出样本值:\n", output[0])
