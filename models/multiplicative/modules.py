import torch
import torch.nn as nn

class MultiplicativeFusion(nn.Module):
    def __init__(
            self, 
            in1_features: int,
            in2_features: int,
            out_features: int,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.bil = nn.Bilinear(in1_features=in1_features,
                               in2_features=in2_features,
                               out_features=out_features,
                               bias=True)
        self.lin1 = nn.Linear(in_features=in1_features,
                              out_features=out_features,
                              bias=False)
        self.lin2 = nn.Linear(in_features=in2_features,
                              out_features=out_features,
                              bias=False)
        
    def forward(
            self,
            in1: torch.Tensor,
            in2: torch.Tensor):
        return self.bil(in1, in2) + self.lin1(in1) + self.lin2(in2)


class MultiInputSequential(nn.Sequential):
    def __init__(
            self,
            *modules: nn.Module):
        self.module_list = [*modules]

    def forward(
            self,
            *input: torch.Tensor):
        x = input
        for module in self.module_list:
            x = module(*x)
        return x