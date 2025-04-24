import torch
import torch.nn as nn



class LayerNorm(nn.Module):
    def __init__(self, feature_size, epsilon=1e-9):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_size))
        self.beta = nn.Parameter(torch.zeros(feature_size))
        self.epsilon = epsilon

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

if __name__ == "__main__":
    # 创建一个 LayerNorm 实例
    layer_norm = LayerNorm(feature_size=4)

    # 输入一个张量，形状为 (batch_size=2, feature_size=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]])

    # 应用 LayerNorm
    output = layer_norm(x)

    print("输入：\n", x)
    print("归一化输出：\n", output)