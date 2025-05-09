import argparse, os, random, time, warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cma

from VAE import VAE
from RNN_MDN import RNN_MDN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("[info] loading VAE …")
vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device).eval()
vae.load_state_dict(torch.load("vae_weights_epoch_02.pth", map_location=device, weights_only=True))

print("[info] loading RNN …")
rnn = RNN_MDN(input_size=32,
              action_dim=3,
              hidden_size=35,
              num_gaussians=5,
              hidden_layer=256,
              num_layers=1).to(device).eval()
rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_40.pth", map_location=device, weights_only=True))

# ---------------------------------------------------------------------------
# Controller definition (z ⊕ h → action)
# ---------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, z_dim: int, h_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + h_dim, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat((z, h), dim=1)
        x = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(x))
        steer = out[:, 0]
        gas   = torch.sigmoid(out[:, 1])
        brake = torch.sigmoid(out[:, 2])
        return torch.stack([steer, gas, brake], dim=1)

controller = Controller(32, 35).to(device)

# helpers to flatten / restore params

def params_to_vec(net: nn.Module) -> np.ndarray:
    return torch.cat([p.data.flatten() for p in net.parameters()]).cpu().numpy()

def vec_to_params(net: nn.Module, vec: np.ndarray):
    idx = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(vec[idx:idx+n], dtype=p.data.dtype).view_as(p))
        idx += n

param_dim = len(params_to_vec(controller))
print(f"[info] controller param dim = {param_dim}")

# ---------------------------------------------------------------------------
# Real‑environment rollout
# ---------------------------------------------------------------------------
ENV_NAME = "CarRacing-v2"

@torch.no_grad()
def evaluate(param: np.ndarray, seed: int = 0, max_steps: int = 1000) -> float:
    env = gym.make(ENV_NAME, render_mode=None)
    env.reset(seed=seed)
    vec_to_params(controller, param)

    obs, _ = env.reset()
    h, c = rnn.get_initial_hidden(device, 1)
    total_reward = 0.0

    for _ in range(max_steps):
        # encode observation → latent μ
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_t = obs_t.permute(2,0,1).unsqueeze(0) / 255.0
        z_mu, _ = vae.encode(obs_t)

        # controller chooses action using current h
        action_t = controller(z_mu, h[0]).cpu().numpy()[0]
        obs, reward, terminated, truncated, _ = env.step(action_t)
        total_reward += reward

        # update hidden state via RNN (teacher forcing with real z)
        ( _ , (h, c)) = rnn(z_mu, (h, c), torch.tensor(action_t, device=device).unsqueeze(0))

        if terminated or truncated:
            break

    env.close()
    return total_reward

# ---------------------------------------------------------------------------
# CMA‑ES optimisation loop
# ---------------------------------------------------------------------------
print("[info] starting CMA‑ES …")
x0 = params_to_vec(controller)
es = cma.CMAEvolutionStrategy(x0, 0.5,  
                              {"popsize": 32, "seed": 42})

for gen in range(80):
    solutions = es.ask()
    fitnesses = [-evaluate(sol, seed=gen*123 + i) for i, sol in enumerate(solutions)]
    es.tell(solutions, fitnesses)
    es.disp()
    print(f"[gen {gen:03d}]  best reward so far ≈ {-es.best.f:.1f}")

print("[done] best solution in real env reward:", -es.result.fbest)
np.save("cma_best_real.npy", es.best.x)
print("weights saved → cma_best_real.npy")