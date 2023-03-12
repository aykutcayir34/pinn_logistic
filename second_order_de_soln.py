"""
d^2y
---- + p(x).dy/dx + q(x).y = f(x)
dx^2
x in [0, 1]
y(0) = A
y(1) = B
p(x) = 0
q(x) = 0
f(x) = -1
y(0) = 0
y(1) = 0
y(x) = (-1/2)x^2 + (1/2)x 
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.out_layer = nn.Linear(10, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.hidden_layer(x))
        return self.out_layer(out)
    
device = "cpu"    
N = MLP().to(device)

def f(x):
    return -torch.ones(x.shape[0], x.shape[1])

def loss(x):
    x.requires_grad = True
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    ddy_ddx = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
    l = torch.mean( (ddy_ddx - f(x))**2 ) + 0.5 * (y[0, 0] - 0.)**2 + 0.5 * (y[-1, 0] - 0.0)**2
    return l

x = torch.linspace(0, 1, 100)[:, None]
optimizer = torch.optim.LBFGS(N.parameters())

def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

epochs = 10
for i in range(epochs):
    optimizer.step(closure)

xx = torch.linspace(0, 1, 100)[:, None]
with torch.no_grad():
    yy = N(xx)

plt.figure(figsize=(10, 6))
plt.plot(xx, yy, label="Predicted")
plt.plot(xx, -0.5 * torch.pow(xx, 2) + 0.5 * xx, "--", label="Exact")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("second_order_diff_eqn_soln.png")
plt.show()