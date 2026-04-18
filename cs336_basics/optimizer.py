from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from einops import reduce

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m = state['m']
                v = state['v']
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta1 * m + (1 - beta1) * grad # update the first moment estimate
                v = beta2 * v + (1 - beta2) * (grad ** 2) # update the second moment estimate
                lr_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                p.data -= lr_t / (torch.sqrt(v) + eps) * m # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state['m'] = m # update m
                state['v'] = v # update v
        return loss

def learning_rate_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int):
    if t < T_w:
        lr_t = t / T_w * lr_max
    if T_w <= t and t <= T_c:
        lr_t = lr_min + 0.5 * (1 + math.cos((t-T_w)/(T_c-T_w)*math.pi))*(lr_max-lr_min)
    if t > T_c:
        lr_t = lr_min
    return lr_t

def gradient_clipping(params: list[torch.Tensor], max_l2_norm: float, eps: float = 1e-6):
    grads = [param.grad for param in params if param.grad is not None]
    if grads is None:
        return
    grads_norm = math.sqrt(sum(torch.sum(grad**2) for grad in grads))
    if grads_norm > max_l2_norm:
        scale_factor = max_l2_norm / (grads_norm + eps)
        for grad in grads:
            grad.mul_(scale_factor)
