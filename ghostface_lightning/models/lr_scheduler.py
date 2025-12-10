"""
Learning Rate Scheduler for GhostFaceNets
"""

import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRWarmup(_LRScheduler):
    """
    Polynomial learning rate scheduler with warmup
    Used in ArcFace training
    """

    def __init__(
        self,
        optimizer,
        warmup_iters,
        total_iters=5,
        power=1.0,
        last_epoch=-1,
    ):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = (
                (1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))
            ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            return [
                (
                    base_lr
                    * (
                        1.0
                        - (min(self.total_iters, self.last_epoch) - self.warmup_iters)
                        / (self.total_iters - self.warmup_iters)
                    )
                    ** self.power
                )
                for base_lr in self.base_lrs
            ]


class WarmupLR(_LRScheduler):
    """
    Wrapper for learning rate scheduler with warmup
    """

    def __init__(self, scheduler, warmup_iters, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        # Store initial learning rates BEFORE calling super().__init__
        # because super().__init__ will call _initial_step() which calls get_lr()
        self.initial_lrs = [group["lr"] for group in scheduler.optimizer.param_groups]
        # Initialize with scheduler's optimizer to get base_lrs
        super().__init__(scheduler.optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: linear warmup
            # Use self.initial_lrs if available, otherwise use self.base_lrs
            base_lrs = getattr(self, "initial_lrs", self.base_lrs)
            return [
                initial_lr * (self.last_epoch + 1) / self.warmup_iters
                for initial_lr in base_lrs
            ]
        else:
            # Use wrapped scheduler
            return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_iters:
            super().step(epoch)
        else:
            self.scheduler.step(epoch)
            self.last_epoch = self.scheduler.last_epoch
