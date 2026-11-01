from torch.optim.lr_scheduler import CosineAnnealingLR


def cosine_lr(optimizer, total_steps, eta_min):
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
