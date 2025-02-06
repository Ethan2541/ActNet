import torch
import torch.nn as nn

from ActLayer import ActLayer

class ActNet(nn.Module):
    """ActNet model.
    
    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        d (int): Latent space dimension.
        m (int): Hidden dimension of the ActLayer modules.
        N (int): Number of basis functions used in each layer (basis size).
        L (int): Number of ActLayer modules in the network.
        w0 (float): Multiplicative coefficient for the original input. High
            values of w0 help approximating highly oscillatory functions,
            while low values are better for smooth functions (cf. Appendix D.1).
        eps (float): A small value to avoid numerical instability.
        init (str): Initialization method used for the parameters.
        bias (bool): Whether to include a bias term in the network.
    """
    def __init__(self, d_in, d_out, d, m, N, L, w0=1.0, init="uniform", bias=False):
        super(ActNet, self).__init__()

        hidden_dims = [d] + [m] * (L-1)

        self.w0 = w0
        self.input_projection = nn.Linear(d_in, d, bias=bias)
        self.act_layers = nn.Sequential(*[ActLayer(hidden_dims[i], m, N, init, bias) for i in range(L-1)])
        self.output_projection = nn.Linear(hidden_dims[-1], d_out, bias=bias)


    def forward(self, x):
        x = self.w0 * x
        x = self.input_projection(x)
        x = self.act_layers(x)
        x = self.output_projection(x)
        return x