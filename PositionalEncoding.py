import torch
import torch.nn as nn
import math




class PositionalEncoding(nn.Module):

     def __init__(self, dim, dropout, max_len=5000):
         super().__init__()
         self.dropout = nn.Dropout(p=dropout)

         # 输入序列的每一个token都有一个位置向量和他对应
         pe = torch.zeros(max_len, dim)
         # sin，cos函数对应的除数都是一样的，
         div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
         # token的位置
         self.position = torch.arange(0, max_len).unsqueeze(1)
         # self.pos和self.pe都拿出同一行，div_term是所需位置向量长度的一半，一次填充一个向量的一半
         pe[:, 0::2] = torch.sin(self.position * div_term)
         pe[:, 1::2] = torch.cos(self.position * div_term)
         pe = pe.unsqueeze(0)

         # 将位置编码注册为模型的缓冲区，不作为参数更新
         self.register_buffer('pe', pe)


     def forward(self, x):
         ## x.shape(batchsize, seqlen, dim)

         x = x + self.pe[:, :x.size(1), :]
         return self.dropout(x)
