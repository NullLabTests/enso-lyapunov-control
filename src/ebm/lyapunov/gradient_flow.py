import torch
from tqdm import tqdm
from ebm.model.jepa import SudokuJEPA

@torch.no_grad()
def solve_with_gradient_flow(model: SudokuJEPA, puzzle, mask, steps: int = 80, eta: float = 0.06):
    model.eval()
    device = next(model.parameters()).device
    
    z = torch.randn(puzzle.shape[0], model.latent_dim, device=device) * 0.1
    energies = []

    for _ in tqdm(range(steps)):
        z = z.clone().requires_grad_(True)
        energy = model.energy(z, model.get_context(puzzle)) + model.constraint_penalty(z)
        energy.mean().backward()
        grad = z.grad
        z = (z - eta * grad).detach()
        energies.append(energy.mean().item())

    solution = model.decode(puzzle, mask, z)
    return solution, energies
