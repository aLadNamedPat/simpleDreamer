# train_controller_cma_parallel.py

"""
Parallel CMA-ES Controller Training (VAE-only, Linear Controller)

Based on the World Models paper approach:
- Linear controller: z -> action (no hidden layers)
- Multiprocessing for parallel evaluation
- Lower sigma (0.1) as in reference implementation

This version uses only VAE latents (no RNN) to first verify
the VAE representations are useful.
"""

import argparse
import sys
import os
from os.path import join, exists
from os import mkdir, getpid
import time
from time import sleep
import torch
import torch.nn as nn
from torch.multiprocessing import Process, Queue, set_start_method
import cma
import numpy as np
import gymnasium as gym
from tqdm import tqdm

# Must be called before any other multiprocessing code
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # Already set

from VAE import VAE


# =============================================================================
# Linear Controller (matches World Models paper)
# =============================================================================

class LinearController(nn.Module):
    """
    Simple linear controller: z -> action
    
    This is what the original World Models paper used.
    The idea is that if VAE/RNN learn good representations,
    a linear mapping should be sufficient.
    """
    
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, action_dim)
    
    def forward(self, x):
        return torch.tanh(self.fc(x))
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def flatten_parameters(model):
    """Flatten model parameters to a 1D numpy array."""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()


def load_parameters(params, model):
    """Load parameters from a 1D numpy array into model."""
    params_tensor = torch.from_numpy(params).float()
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx + size].view(p.shape))
        idx += size


# =============================================================================
# Rollout Generator (runs in worker processes)
# =============================================================================

class RolloutGenerator:
    """
    Generates rollouts for evaluation.
    Each worker process has its own instance.
    """
    
    def __init__(self, vae_weights_path: str, latent_dim: int, device: torch.device, time_limit: int = 1000):
        self.device = device
        self.time_limit = time_limit
        self.latent_dim = latent_dim
        
        # Create environment
        self.env = gym.make("CarRacing-v3", continuous=True)
        
        # Load VAE
        self.vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(device)
        self.vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
        self.vae.eval()
        
        # Create controller (weights will be set per rollout)
        self.controller = LinearController(latent_dim, 3).to(device)
    
    def rollout(self, params: np.ndarray) -> float:
        """
        Execute one rollout with given controller parameters.
        
        Args:
            params: Flattened controller parameters
        
        Returns:
            Negative cumulative reward (for CMA-ES minimization)
        """
        # Load parameters into controller
        load_parameters(params, self.controller)
        
        with torch.no_grad():
            obs, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(self.time_limit):
                # Encode observation
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                mu, _ = self.vae.encode(obs_tensor)
                z = mu  # Use mean for deterministic behavior
                
                # Get action
                action = self.controller(z)
                a = action.squeeze(0).cpu().numpy()
                
                # Scale to CarRacing action space
                # steering: [-1, 1] (already correct)
                # gas: [-1, 1] -> [0, 1]
                # brake: [-1, 1] -> [0, 1]
                a[1] = (a[1] + 1) / 2
                a[2] = (a[2] + 1) / 2
                a = a.astype(np.float32)
                
                # Step environment
                obs, reward, terminated, truncated, _ = self.env.step(a)
                total_reward += reward
                
                if terminated or truncated:
                    break
        
        # Return negative reward (CMA-ES minimizes)
        return -total_reward
    
    def close(self):
        self.env.close()


# =============================================================================
# Worker Process
# =============================================================================

def worker_routine(
    p_queue: Queue,      # Parameters queue (input)
    r_queue: Queue,      # Results queue (output)
    e_queue: Queue,      # End signal queue
    worker_id: int,
    vae_weights_path: str,
    latent_dim: int,
    time_limit: int,
    tmp_dir: str,
):
    """
    Worker process routine.
    
    Pulls (solution_id, params) from p_queue, executes rollout,
    pushes (solution_id, reward) to r_queue.
    
    Terminates when e_queue is non-empty.
    """
    # Determine device (distribute across GPUs if available)
    if torch.cuda.is_available():
        gpu_id = worker_id % torch.cuda.device_count()
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    # Redirect stdout/stderr to files (optional, helps with debugging)
    if tmp_dir:
        sys.stdout = open(join(tmp_dir, f'{getpid()}.out'), 'a')
        sys.stderr = open(join(tmp_dir, f'{getpid()}.err'), 'a')
    
    # Initialize rollout generator
    try:
        r_gen = RolloutGenerator(vae_weights_path, latent_dim, device, time_limit)
        
        while e_queue.empty():
            if p_queue.empty():
                sleep(0.1)
            else:
                try:
                    s_id, params = p_queue.get(timeout=1)
                    result = r_gen.rollout(params)
                    r_queue.put((s_id, result))
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
        
        r_gen.close()
    except Exception as e:
        print(f"Worker {worker_id} initialization error: {e}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train_controller_parallel(
    vae_weights_path: str,
    latent_dim: int = 32,
    pop_size: int = 16,
    n_samples: int = 4,
    max_generations: int = 100,
    target_return: float = 900,
    sigma_init: float = 0.1,
    num_workers: int = None,
    time_limit: int = 1000,
    eval_rollouts: int = 100,
    log_step: int = 3,
    save_dir: str = "ctrl_checkpoints",
    display: bool = True,
):
    """
    Train controller using CMA-ES with parallel evaluation.
    
    Args:
        vae_weights_path: Path to trained VAE weights
        latent_dim: VAE latent dimension
        pop_size: CMA-ES population size
        n_samples: Number of rollouts per candidate for fitness estimation
        max_generations: Maximum generations
        target_return: Stop if return exceeds this
        sigma_init: Initial CMA-ES step size
        num_workers: Number of worker processes (default: n_samples * pop_size)
        time_limit: Max steps per rollout
        eval_rollouts: Number of rollouts for periodic evaluation
        log_step: Evaluate best every log_step generations
        save_dir: Directory to save checkpoints
        display: Show progress bars
    """
    
    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = join(save_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(32, n_samples * pop_size)
    
    print("=" * 60)
    print("Parallel CMA-ES Controller Training")
    print("=" * 60)
    print(f"Population size: {pop_size}")
    print(f"Samples per candidate: {n_samples}")
    print(f"Number of workers: {num_workers}")
    print(f"Sigma init: {sigma_init}")
    print(f"Target return: {target_return}")
    print(f"Time limit: {time_limit}")
    print("=" * 60)
    
    # Create queues
    p_queue = Queue()  # Parameters
    r_queue = Queue()  # Results
    e_queue = Queue()  # End signal
    
    # Start worker processes
    print(f"Starting {num_workers} worker processes...")
    workers = []
    for worker_id in range(num_workers):
        p = Process(
            target=worker_routine,
            args=(p_queue, r_queue, e_queue, worker_id, 
                  vae_weights_path, latent_dim, time_limit, tmp_dir)
        )
        p.start()
        workers.append(p)
    
    # Give workers time to initialize
    print("Waiting for workers to initialize...")
    sleep(5)
    
    # Create dummy controller to get parameter count
    controller = LinearController(latent_dim, 3)
    num_params = controller.get_num_params()
    print(f"Controller parameters: {num_params}")
    
    # Initialize CMA-ES
    x0 = flatten_parameters(controller)
    es = cma.CMAEvolutionStrategy(x0, sigma_init, {'popsize': pop_size})
    
    # Track best
    cur_best = None
    best_params = None
    
    # Check for existing checkpoint
    ctrl_file = join(save_dir, 'best.tar')
    if exists(ctrl_file):
        print(f"Loading existing checkpoint from {ctrl_file}")
        state = torch.load(ctrl_file, map_location='cpu')
        cur_best = -state['reward']  # Stored as positive, we track negative
        controller.load_state_dict(state['state_dict'])
        best_params = flatten_parameters(controller)
        print(f"Previous best: {-cur_best:.1f}")
    
    # Training loop
    generation = 0
    try:
        while not es.stop() and generation < max_generations:
            # Check if we've reached target
            if cur_best is not None and -cur_best > target_return:
                print(f"Reached target return {target_return}, stopping!")
                break
            
            # Get candidate solutions
            solutions = es.ask()
            
            # Push all evaluations to queue
            # Each solution is evaluated n_samples times
            for s_id, solution in enumerate(solutions):
                for _ in range(n_samples):
                    p_queue.put((s_id, solution))
            
            # Collect results
            r_list = [0.0] * pop_size
            total_evals = pop_size * n_samples
            
            if display:
                pbar = tqdm(total=total_evals, desc=f"Gen {generation + 1}")
            
            collected = 0
            while collected < total_evals:
                if not r_queue.empty():
                    s_id, result = r_queue.get()
                    r_list[s_id] += result / n_samples  # Average over samples
                    collected += 1
                    if display:
                        pbar.update(1)
                else:
                    sleep(0.05)
            
            if display:
                pbar.close()
            
            # Tell CMA-ES the results
            es.tell(solutions, r_list)
            
            # Print generation stats
            rewards = [-r for r in r_list]  # Convert back to positive rewards
            gen_mean = np.mean(rewards)
            gen_max = np.max(rewards)
            gen_min = np.min(rewards)
            
            print(f"Gen {generation + 1:3d} | "
                  f"Mean: {gen_mean:7.1f} | "
                  f"Max: {gen_max:7.1f} | "
                  f"Min: {gen_min:7.1f} | "
                  f"Sigma: {es.sigma:.4f}")
            
            # Periodic evaluation with more rollouts
            if generation % log_step == log_step - 1:
                print(f"\nEvaluating best candidate with {eval_rollouts} rollouts...")
                
                # Find best from this generation
                best_idx = np.argmin(r_list)
                eval_params = solutions[best_idx]
                
                # Push evaluation rollouts
                for s_id in range(eval_rollouts):
                    p_queue.put((s_id, eval_params))
                
                # Collect evaluation results
                eval_results = []
                if display:
                    pbar = tqdm(total=eval_rollouts, desc="Evaluating")
                
                while len(eval_results) < eval_rollouts:
                    if not r_queue.empty():
                        _, result = r_queue.get()
                        eval_results.append(-result)  # Convert to positive
                        if display:
                            pbar.update(1)
                    else:
                        sleep(0.05)
                
                if display:
                    pbar.close()
                
                eval_mean = np.mean(eval_results)
                eval_std = np.std(eval_results)
                
                print(f"Evaluation: {eval_mean:.1f} ± {eval_std:.1f}")
                
                # Save if best
                if cur_best is None or -eval_mean < cur_best:
                    cur_best = -eval_mean
                    best_params = eval_params.copy()
                    
                    # Save checkpoint
                    load_parameters(best_params, controller)
                    torch.save({
                        'generation': generation,
                        'reward': eval_mean,
                        'reward_std': eval_std,
                        'state_dict': controller.state_dict(),
                    }, ctrl_file)
                    
                    print(f"New best! Saved to {ctrl_file}")
                
                print(f"Best ever: {-cur_best:.1f}\n")
            
            generation += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Signal workers to stop
        print("Shutting down workers...")
        e_queue.put('STOP')
        
        # Wait for workers to finish
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        print("Workers stopped")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    if cur_best is not None:
        print(f"Best reward: {-cur_best:.1f}")
        print(f"Saved to: {ctrl_file}")
    
    return -cur_best if cur_best else None


# =============================================================================
# Single-threaded evaluation for testing
# =============================================================================

def evaluate_single(vae_weights_path: str, controller_path: str, latent_dim: int = 32, 
                    num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained controller (single-threaded, for testing).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VAE
    vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(device)
    vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    vae.eval()
    
    # Load controller
    controller = LinearController(latent_dim, 3).to(device)
    state = torch.load(controller_path, map_location=device)
    controller.load_state_dict(state['state_dict'])
    controller.eval()
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        with torch.no_grad():
            for _ in range(1000):
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                mu, _ = vae.encode(obs_tensor)
                action = controller(mu)
                
                a = action.squeeze(0).cpu().numpy()
                a[1] = (a[1] + 1) / 2
                a[2] = (a[2] + 1) / 2
                a = a.astype(np.float32)
                
                obs, reward, terminated, truncated, _ = env.step(a)
                total_reward += reward
                
                if terminated or truncated:
                    break
        
        rewards.append(total_reward)
        print(f"Episode {ep + 1}: {total_reward:.1f}")
    
    env.close()
    
    print(f"\nMean: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Max: {np.max(rewards):.1f}, Min: {np.min(rewards):.1f}")
    
    return rewards


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel CMA-ES Controller Training")
    
    # Paths
    parser.add_argument("--vae_weights", type=str, default="vae_weights_epoch_04.pth",
                        help="Path to trained VAE weights")
    parser.add_argument("--save_dir", type=str, default="ctrl_checkpoints",
                        help="Directory to save checkpoints")
    
    # Model
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="VAE latent dimension")
    
    # CMA-ES
    parser.add_argument("--pop_size", type=int, default=16,
                        help="Population size")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Rollouts per candidate for fitness")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Initial CMA-ES step size")
    parser.add_argument("--generations", type=int, default=100,
                        help="Maximum generations")
    parser.add_argument("--target_return", type=float, default=900,
                        help="Target return to stop training")
    
    # Workers
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes")
    parser.add_argument("--time_limit", type=int, default=1000,
                        help="Max steps per rollout")
    
    # Evaluation
    parser.add_argument("--eval_rollouts", type=int, default=100,
                        help="Rollouts for periodic evaluation")
    parser.add_argument("--log_step", type=int, default=3,
                        help="Evaluate every N generations")
    
    # Display
    parser.add_argument("--no_display", action="store_true",
                        help="Disable progress bars")
    
    # Eval mode
    parser.add_argument("--eval", action="store_true",
                        help="Evaluation mode (don't train)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Episodes for evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        # Evaluation mode
        ctrl_file = join(args.save_dir, 'best.tar')
        if not exists(ctrl_file):
            print(f"No checkpoint found at {ctrl_file}")
            return
        
        evaluate_single(
            args.vae_weights,
            ctrl_file,
            args.latent_dim,
            args.eval_episodes,
            args.render,
        )
    else:
        # Training mode
        train_controller_parallel(
            vae_weights_path=args.vae_weights,
            latent_dim=args.latent_dim,
            pop_size=args.pop_size,
            n_samples=args.n_samples,
            max_generations=args.generations,
            target_return=args.target_return,
            sigma_init=args.sigma,
            num_workers=args.num_workers,
            time_limit=args.time_limit,
            eval_rollouts=args.eval_rollouts,
            log_step=args.log_step,
            save_dir=args.save_dir,
            display=not args.no_display,
        )


if __name__ == "__main__":
    main()