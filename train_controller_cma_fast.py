# train_controller_cma_fast.py

import torch
import numpy as np
import gymnasium as gym
import cma
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_controller_vectorized(vae, rnn, controller, vec_env, max_steps=1000):
    """
    Evaluate controller across multiple environments in parallel.
    """
    num_envs = vec_env.num_envs
    
    vae.eval()
    rnn.eval()
    controller.eval()
    
    with torch.no_grad():
        obs, _ = vec_env.reset()
        h = rnn.get_initial_hidden(device, batch_size=num_envs)
        
        total_rewards = np.zeros(num_envs)
        dones = np.zeros(num_envs, dtype=bool)
        steps = 0
        
        while not np.all(dones) and steps < max_steps:
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            h_for_controller = h[0][-1]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            actions = controller(controller_input)
            actions = torch.tanh(actions)
            
            a = actions.cpu().numpy()
            a[:, 1] = (a[:, 1] + 1) / 2
            a[:, 2] = (a[:, 2] + 1) / 2
            a = a.astype(np.float32)
            
            obs, rewards, terminated, truncated, _ = vec_env.step(a)
            done_now = terminated | truncated
            
            total_rewards += rewards * (~dones)
            dones = dones | done_now
            
            a_tensor = torch.from_numpy(a).float().to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            steps += 1
        
        return total_rewards


# Global models for worker processes
_worker_models = {}

def init_worker(vae_path, rnn_path, latent_dim, hidden_size, eval_episodes):
    """Initialize models in worker process."""
    global _worker_models
    
    # CPU for workers to avoid GPU memory issues
    worker_device = torch.device('cpu')
    
    vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(worker_device)
    rnn = RNN_MDN(latent_dim, 3, hidden_size, 5, 256, 1).to(worker_device)
    controller = Controller(input_features=latent_dim + hidden_size, actions_dims=3).to(worker_device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=worker_device))
    rnn.load_state_dict(torch.load(rnn_path, map_location=worker_device))
    
    vae.eval()
    rnn.eval()
    
    # Create vectorized env for this worker
    vec_env = gym.vector.AsyncVectorEnv([
        lambda: gym.make("CarRacing-v3") for _ in range(eval_episodes)
    ])
    
    _worker_models = {
        'vae': vae,
        'rnn': rnn,
        'controller': controller,
        'vec_env': vec_env,
        'device': worker_device,
    }


def evaluate_candidate_worker(args):
    """Evaluate a single candidate in worker process."""
    params, max_steps = args
    global _worker_models
    
    vae = _worker_models['vae']
    rnn = _worker_models['rnn']
    controller = _worker_models['controller']
    vec_env = _worker_models['vec_env']
    device = _worker_models['device']
    
    # Set controller params
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size
    
    num_envs = vec_env.num_envs
    
    with torch.no_grad():
        obs, _ = vec_env.reset()
        h = rnn.get_initial_hidden(device, batch_size=num_envs)
        
        total_rewards = np.zeros(num_envs)
        dones = np.zeros(num_envs, dtype=bool)
        steps = 0
        
        while not np.all(dones) and steps < max_steps:
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            h_for_controller = h[0][-1]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            actions = controller(controller_input)
            actions = torch.tanh(actions)
            
            a = actions.cpu().numpy()
            a[:, 1] = (a[:, 1] + 1) / 2
            a[:, 2] = (a[:, 2] + 1) / 2
            a = a.astype(np.float32)
            
            obs, rewards, terminated, truncated, _ = vec_env.step(a)
            done_now = terminated | truncated
            
            total_rewards += rewards * (~dones)
            dones = dones | done_now
            
            a_tensor = torch.from_numpy(a).float().to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            steps += 1
        
        return np.mean(total_rewards)


def get_controller_params(controller):
    return torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()


def set_controller_params(controller, params):
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size


def train_controller_cma_parallel(
    vae_path,
    rnn_path,
    latent_dim=32,
    hidden_size=256,
    max_generations=100,
    population_size=64,
    sigma_init=0.5,
    eval_episodes=16,
    max_steps=1000,
    num_workers=8,
):
    """
    Train controller using CMA-ES with fully parallel evaluation.
    
    Each worker has its own vectorized environment for evaluating candidates.
    """
    print(f"Starting parallel training with {num_workers} workers")
    print(f"Each worker runs {eval_episodes} parallel envs")
    print(f"Total parallel envs: {num_workers * eval_episodes}")
    
    # Load models on main process for initialization
    controller = Controller(input_features=latent_dim + hidden_size, actions_dims=3).to('cpu')
    x0 = get_controller_params(controller)
    num_params = len(x0)
    
    print(f"Training controller with {num_params} parameters")
    print(f"Population: {population_size}, Eval episodes: {eval_episodes}")
    
    # CMA-ES options
    opts = {
        'popsize': population_size,
        'maxiter': max_generations,
        'CMA_diagonal': True,
        'verb_disp': 0,
        'verb_log': 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma_init, opts)
    
    best_reward = -float('inf')
    best_params = x0.copy()
    generation = 0
    
    # Create process pool with initialized workers
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(vae_path, rnn_path, latent_dim, hidden_size, eval_episodes)
    ) as executor:
        
        try:
            while not es.stop():
                candidates = es.ask()
                
                # Submit all candidates for parallel evaluation
                args_list = [(c.astype(np.float32), max_steps) for c in candidates]
                
                # Evaluate all candidates in parallel
                rewards = list(tqdm(
                    executor.map(evaluate_candidate_worker, args_list),
                    total=len(candidates),
                    desc=f"Gen {generation}",
                    leave=False
                ))
                
                # CMA-ES minimizes
                fitnesses = [-r for r in rewards]
                es.tell(candidates, fitnesses)
                
                # Track best
                gen_best_idx = np.argmin(fitnesses)
                gen_best_reward = rewards[gen_best_idx]
                
                if gen_best_reward > best_reward:
                    best_reward = gen_best_reward
                    best_params = candidates[gen_best_idx].copy()
                    
                    # Save best
                    controller_save = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
                    set_controller_params(controller_save, best_params)
                    torch.save(controller_save.state_dict(), "controller_cma_best.pth")
                
                print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards):7.1f} | "
                      f"Max: {np.max(rewards):7.1f} | Best ever: {best_reward:7.1f} | "
                      f"Sigma: {es.sigma:.4f}")
                
                # Checkpoint every 10 generations
                if (generation + 1) % 10 == 0:
                    controller_save = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
                    set_controller_params(controller_save, best_params)
                    torch.save(controller_save.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
                
                generation += 1
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
    
    print(f"\nCMA-ES stopped: {es.stop()}")
    return best_reward, best_params


def main():
    parser = argparse.ArgumentParser(description="Train controller with CMA-ES (Fully Parallel)")
    parser.add_argument("--vae", default="vae_weights_epoch_07.pth", help="VAE weights path")
    parser.add_argument("--rnn", default="weights_rnn/RNN_weights_epoch_02.pth", help="RNN weights path")
    parser.add_argument("--generations", type=int, default=100, help="Max generations")
    parser.add_argument("--population", type=int, default=64, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial sigma")
    parser.add_argument("--eval-episodes", type=int, default=16, help="Episodes per evaluation (per worker)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()
    
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), args.population)
    
    print("=" * 60)
    print("CMA-ES Controller Training (Fully Parallel)")
    print("=" * 60)
    print(f"VAE: {args.vae}")
    print(f"RNN: {args.rnn}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num workers: {args.num_workers}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print("=" * 60)
    
    latent_dim = 32
    hidden_size = 256
    
    best_reward, best_params = train_controller_cma_parallel(
        vae_path=args.vae,
        rnn_path=args.rnn,
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        max_generations=args.generations,
        population_size=args.population,
        sigma_init=args.sigma,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
    )
    
    # Final save
    controller_final = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
    params_tensor = torch.from_numpy(best_params).float()
    idx = 0
    for p in controller_final.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size
    
    torch.save(controller_final.state_dict(), "controller_cma_final.pth")
    print(f"\nTraining complete! Best reward: {best_reward:.1f}")
    print("Saved: controller_cma_final.pth, controller_cma_best.pth")


if __name__ == "__main__":
    main()