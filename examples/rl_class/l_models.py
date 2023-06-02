import torch

class ConstantL(torch.nn.Module):
    def __init__(self, max_constraint=0.0):
        super().__init__()
        self.l = torch.nn.Parameter(torch.rand(1))
        self.t = max_constraint

    def forward(self, a, c):
        return self.l * (c - self.t)


class LinearL(torch.nn.Module):

    def __init__(self, in_features, bias=False, max_constraint=0.0):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1, bias)
        self.t = max_constraint

    def forward(self, a, c):
        l = self.linear(a)
        return l * (c - self.t)



def get_l_models(lambda_type, max_constraint, in_features=1):
    if lambda_type=='linear':
        return LinearL(in_features, max_constraint=max_constraint)
    else:
        return ConstantL(max_constraint=max_constraint)
