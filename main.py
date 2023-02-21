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

def f(nn: LogisticPINN, tf: torch.Tensor) -> torch.Tensor:
    return nn(tf)

def df(nn: LogisticPINN, tf: torch.Tensor, order: int = 1) -> torch.Tensor:
    d_nn_eval = nn(tf)
    for d in range(order):
        d_nn_eval = torch.autograd.grad(
            d_nn_eval, tf, grad_outputs=torch.ones_like(tf), create_graph=True, retain_graph=True
        )[0]
    return d_nn_eval

def loss_function(nn: LogisticPINN, tf: torch.Tensor) -> torch.Tensor:
    interior_loss = df(nn, tf, order=1) - R * tf * (1 - tf)
    boundary = torch.Tensor([T0])
    boundary.requires_grad = True
    boundary_loss = f(nn, boundary) - F0
    total_loss = interior_loss.pow(2).mean() + boundary_loss**2
    return total_loss

def train(nn: LogisticPINN, tf: torch.Tensor) -> torch.Tensor:
    epochs = 20_000
    lr = 1e-2
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(nn, tf)
        loss_list = []
        loss_list.append(loss.item())
        loss.backward() 
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    return nn, loss_list

if __name__ == "__main__":
    set_seed()
    t_u = torch.linspace(0, 1, 100, requires_grad=True)
    t_u = t_u.reshape(t_u.shape[0], 1)
    tf = torch.linspace(0, 1, 10, requires_grad=True)
    tf = tf.reshape(tf.shape[0], 1)

    T0 = 0
    F0 = 0
    R = 1

    nn = LogisticPINN()
    nn_trained, lost_list = train(nn, tf)
    fig, ax = plt.subplots()
    f_final_training = f(nn_trained, tf)
    f_final = f(nn_trained, t_u)

    ax.scatter(tf.detach().numpy(), f_final_training.detach().numpy(), label="Training Points", color="red")
    ax.plot(t_u.detach().numpy(), f_final.detach().numpy(), label="NN Final Solution")
    plt.xlabel(r't')
    plt.ylabel(r'$f(x)$')
    plt.title("Logistic Equation")
    plt.savefig("logistic_pinn.png")
    plt.show()