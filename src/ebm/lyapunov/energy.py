import torch
from torch import Tensor

def lyapunov_candidate(z: Tensor, z_target: Tensor) -> Tensor:
    return ((z - z_target) ** 2).sum(dim=-1)

def gradient_flow_step(z: Tensor, z_context: Tensor, model, eta: float = 0.05):
    z = z.clone().detach().requires_grad_(True)
    energy = model.energy(z, z_context) + model.constraint_penalty(z)
    energy.mean().backward()
    grad = z.grad
    return (z - eta * grad).detach()
