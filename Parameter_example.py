import torch
from torch import nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # (out_features X in_features)
        self.W = Parameter(torch.ones(out_features, in_features))
        self.b = Parameter(torch.ones(out_features))
        print(self.W)
        print(self.b)

    def forward(self, x):

        # addmm : mat1, mat2에 대해 행렬곱 수행
        #
        output = torch.addmm(self.b, x, self.W.T)

        return output


x = torch.Tensor([[1, 2], [3, 4]])

linear = Linear(2, 3)
output = linear(x)

print(output)
