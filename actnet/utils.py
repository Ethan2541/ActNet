import torch

from torch.utils.tensorboard import SummaryWriter


def train(train_data, model, optimizer, scheduler, criterion, n_epochs):
    """Train the model for one epoch.
    
    Args:
        train_data (torch.utils.data.DataLoader): Data loader for training data.
        test_data (torch.utils.data.DataLoader): Data loader for test data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        n_epochs (int): Number of epochs to train the model.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    writer = SummaryWriter()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            adaptive_gradient_clipping(model.parameters())
            optimizer.step()
        scheduler.step()
        
        writer.add_scalar("Loss/Train", total_loss / len(train_data), epoch)
    writer.close()


def adaptive_gradient_clipping(parameters, clip_factor=0.01, eps=1e-3):
    """Adaptive Gradient Clipping (AGC) implementation.
    
    Args:
        parameters (iterable): Iterable of model parameters with gradients to be clipped.
        clip_factor (float): Clipping factor for scaling (default: 0.01 according to Appendix D.2).
        eps (float): Small epsilon to prevent division by zero (default: 1e-3 according to the AGC original paper).
    """

    for param in parameters:
        if param.grad is None:
            continue
        
        param_norm = torch.norm(param)
        grad_norm = torch.norm(param.grad)
        
        if param_norm > 0 and grad_norm > 0:
            scale = clip_factor * param_norm / torch.maximum(grad_norm, torch.tensor(eps))
            
            # Clip gradients if necessary
            if scale < 1.0:
                param.grad.data *= scale


def pinns_loss(u_pred, u_target, f_pred, lambda_u=1.0, lambda_f=1.0):
    loss_u = torch.mean((u_pred - u_target) ** 2)
    loss_f = torch.mean(f_pred ** 2)
    loss = lambda_u * loss_u + lambda_f * loss_f
    return loss