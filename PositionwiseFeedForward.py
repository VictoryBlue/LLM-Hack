import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络。

    参数:
        d_model: 输入和输出向量的维度
        d_ff: FFN 隐藏层的维度，或者说中间层
        dropout: 随机失活率（Dropout），即随机屏蔽部分神经元的输出，用于防止过拟合

    （实际上论文并没有确切地提到在这个模块使用 dropout，所以注释）
    """
    def __init__(self, dim, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(dim, d_ff)
        self.fc2 = nn.Linear(d_ff, dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_positionwise_feedforward():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    d_ff = 64  # 隐藏层升维

    # 输入数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 初始化 FFN 模块
    ffn = PositionwiseFeedForward(d_model, d_ff)

    # 前向传播
    output = ffn(x)

    # 输出检查
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("输出样本值:\n", output[0])

    # 验证输出形状和是否存在 NaN
    assert output.shape == x.shape, "输出维度应该和输入一致"
    assert not torch.isnan(output).any(), "输出中不应有 NaN"

if __name__ == "__main__":
    test_positionwise_feedforward()