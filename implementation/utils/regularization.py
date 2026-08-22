import torch
import torch.nn as nn


def compute_kan_regularization(
    model: nn.Module,
    act_reg: float = 1.0,
    entropy_reg: float = 0.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute L1 and Entropy regularization over network parameters (Eq. 12 in manuscript).
    
    L_reg = act_reg * sum(|p|) + entropy_reg * (-sum(p_bar * log(p_bar)))
    where p_bar = |p| / sum(|p|)
    
    Args:
        model: PyTorch KAN model
        act_reg: Weight for L1 activation regularization
        entropy_reg: Weight for entropy regularization
        eps: Numerical epsilon to prevent log(0)
    """
    params = [p.view(-1) for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.tensor(0.0)
        
    all_p = torch.cat(params)
    l1 = torch.abs(all_p)
    act_loss = torch.sum(l1)
    
    if entropy_reg > 0.0:
        p_bar = l1 / (act_loss + eps)
        entropy_loss = -torch.sum(p_bar * torch.log(p_bar + eps))
        return act_reg * act_loss + entropy_reg * entropy_loss
    else:
        return act_reg * act_loss
