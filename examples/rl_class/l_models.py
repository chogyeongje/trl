import torch

class ConstantL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Parameter(torch.rand(1))
        self.l.data = torch.clamp(self.l.data, min=0)

    def forward(self, a):
        return self.l


class LinearL(torch.nn.Module):

    def __init__(self, in_features, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1, bias)
        self.act = torch.nn.ReLU()

    def forward(self, a):
        print("input shape:", a.shape)
        l = self.linear(a)
        l = self.act(l)
        l = l.sum() / (l!=0).sum()
        return l



def get_l_models(lambda_type, in_features=1):
    if lambda_type=='linear':
        return LinearL(in_features)
    else:
        return ConstantL()
