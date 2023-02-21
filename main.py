import torch
from torch import nn 

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def set_seed(seed=42):
    torch.manual_seed(seed)

class LogisticPINN(nn.Module):
    def __init__(self) -> None:
        super(LogisticPINN, self).__init__()
        self.layer_in = nn.Linear(1, 10)
        self.layer_out = nn.Linear(10, 1)
        self.act = nn.Tanh()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(10, 10) for _ in range(5)]
        )

    def forward(self, t):
        out_0 = self.layer_in(t)
        for hidden_layer in self.hidden_layers:
            out_0 = self.act(hidden_layer(out_0))
        
        return self.layer_out(out_0)

        
