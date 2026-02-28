"""Learning rate scheduling and beta annealing utilities."""

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * alpha for base_lr in self.base_lrs]


def get_beta(epoch: int, beta_start: float, beta_end: float, warmup_epochs: int) -> float:
    """Compute the KL weight (beta) for a given epoch."""
    if warmup_epochs <= 0:
        return beta_end
    return min(beta_end, beta_start + (beta_end - beta_start) * epoch / warmup_epochs)
