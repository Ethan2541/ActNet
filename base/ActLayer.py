import torch
import torch.nn as nn

class ActLayer(nn.Module):
    """Individual Layer of an ActNet.
    
    Args:
        d (int): Input dimension.
        m (int): Hidden dimension.
        N (int): The number of basis functions used in the layer.
        eps (float): A small value to avoid numerical instability.
        init (str): Initialization method used for the parameters.
        bias (bool): Whether to include a bias term in the layer.
    """
    def __init__(self, d, m, N, eps=1e-8, init="uniform", bias=False):
        super(ActLayer, self).__init__()

        # Initialization of the parameters can be either uniform or Gaussian
        # according to Appendix D.1 of the paper
        if init == "uniform":
            self.betas = nn.Parameter(torch.empty(m, N).uniform_(-torch.sqrt(3*N), torch.sqrt(3*N)))
            self.lambdas = nn.Parameter(torch.empty(m, d).uniform_(-torch.sqrt(3*d), torch.sqrt(3*d)))
        elif init == "normal":
            self.betas = nn.Parameter(torch.empty(m, N).normal_(0, torch.sqrt(N)))
            self.lambdas = nn.Parameter(torch.empty(m, d).normal_(0, torch.sqrt(d)))
            
        # The bias is optional
        if bias:
            self.bias = nn.Parameter(torch.zeros(m))
        else:
            self.bias = None

        # The frequencies used for the basis function are drawn from
        # a standard normal distribution, while the phases are constant
        # These parameters are very sensitive to gradient updates, so it
        # is advised to use AGC to train them (cf. Appendix D.2)
        self.frequencies = nn.Parameter(torch.randn(N))
        self.phases = nn.Parameter(torch.zeros(N))
        self.eps = eps
        

    def basis_expansion(self, x):
        mean = torch.exp(-self.frequencies**2 / 2) * torch.sin(self.phases)
        var = 1/2 - torch.exp(-2 * self.frequencies**2) * torch.cos(2 * self.phases) / 2 - mean**2
        std = torch.sqrt(var)
        return (torch.sin(self.frequencies * x + self.phases) - mean) / (std + self.eps)


    def forward(self, x):
        B_x = self.basis_expansion(x)
        out = torch.einsum('ij,jk,ik->k', self.betas, B_x, self.lambdas)
        if self.bias is not None:
            out += self.bias
        return out