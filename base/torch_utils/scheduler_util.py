# -*- coding:utf-8 -*-
# create: 2021/10/9
import math
import torch
from torch.optim import Optimizer
from timm.scheduler.scheduler import Scheduler

from torch.optim.lr_scheduler import LambdaLR


def get_stairs_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    stair_num=2,
                                    min_scale=0.01,
                                    last_epoch=-1,
                                    **kwargs):
    """
    Create a stair schedule with a learning rate that from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    and then with duplicate stairs, more train step will be allocated to a smaller learning rates.
    decrease stage like this with learning rate:4e-4, stair_num:3, remain_steps: 1400
    then learning rate is:
    from 0-100, 4e-4
    from 100-200, decrease from 4e-4 to 2e-4
    from 200-400 2e-4
    from 400-600 decrease from 2e-4 to 1e-4
    from 600-1000 1e-4
    from 1000-1400 decrease from 1e-4 to 0
    as following:
          ___
        /     \ ____
      /              \ ___
    /                      \
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        stair_num: int
        min_scale: min learning_rate ratio
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    stair_num = int(stair_num)
    min_scale = float(min_scale)
    remain_step = max(1, num_training_steps - num_warmup_steps)
    unit_step = int(remain_step / (2**(stair_num + 1) - 2))
    remain_linear_step = remain_step - unit_step * int(2**stair_num - 1)
    stair_steps = [
        unit_step * int((3 * 2**(i / 2) - 2)) if i % 2 == 0 else unit_step * int(4 * 2**((i - 1) / 2) - 2)
        for i in range(2 * stair_num)
    ]
    stair_scales = [1.0 - (2**i - 1) / (2**stair_num - 1) for i in range(stair_num)]

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        current_remain_step = current_step - num_warmup_steps
        for i in range(stair_num * 2):
            if i == 0:
                prev_stair_step = 0
            else:
                prev_stair_step = stair_steps[i - 1]
            stair_step = stair_steps[i]
            if prev_stair_step <= current_remain_step <= stair_step:
                if i % 2 == 0:
                    return max(min_scale, stair_scales[i // 2])
                else:
                    prev_linear_step = unit_step * int(2**((i - 1) / 2) - 1)
                    current_linear_step = current_remain_step - prev_stair_step + prev_linear_step
                    linear_lr = float(remain_linear_step - current_linear_step) / float(remain_linear_step)
                    return max(min_scale, linear_lr)
        return min_scale

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LinearLRScheduler(Scheduler):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min_rate: float,
            warmup_t=0,
            warmup_lr_init=0.,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize=True,
    ) -> None:
        super().__init__(optimizer,
                         param_group_field="lr",
                         noise_range_t=noise_range_t,
                         noise_pct=noise_pct,
                         noise_std=noise_std,
                         noise_seed=noise_seed,
                         initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


# update lr by epoch
def get_cosine_schedule_by_epochs(optimizer: Optimizer, num_epochs: int, last_epoch: int = -1, **kwargs):

    def lr_lambda(epoch):
        lf = ((1 + math.cos(epoch * math.pi / num_epochs)) / 2) * 0.8 + 0.2  # cosine
        return lf

    return LambdaLR(optimizer, lr_lambda, last_epoch)
