import torch
import torch.nn as nn

device = 'cpu'

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.out_layer = nn.Linear(10, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        hidden_out = self.act(self.hidden_layer(x))
        return self.out_layer(hidden_out)
    
N = MLP().to(device)

def f(x):
    return torch.exp(x)

def loss(x):
    x.requires_grad = True
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return torch.mean( (dy_dx - f(x))**2 ) + (y[0, 0] - 1.0)**2

optimizer = torch.optim.LBFGS(N.parameters())
x = torch.linspace(0, 1, 100)[:, None]

def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

epochs = 10
for i in range(epochs):
    optimizer.step(closure)

import matplotlib.pyplot as plt
xx = torch.linspace(0, 1, 100)[:, None]

with torch.no_grad():
    yy = N(xx)

plt.figure(figsize=(10, 6))
plt.plot(xx, yy, label="Predicted")
plt.plot(xx, torch.exp(xx), "--", label="Exact")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("ode_soln.png")
