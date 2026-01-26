# train_controller_cma_moderate.py
"""
Moderately parallel CMA-ES training.
Uses a small number of workers with small vectorized environments.
Designed for systems with ~9GB RAM and 4+ CPU cores.
"""

import torch
import numpy as np
import gymnasium as gym
import cma
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
import argparse
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import gc

# Global models for worker processes
_worker_models = None


def init_worker(vae_path, rnn_path, latent_dim, hidden_size, envs_per_worker):
    """Initialize models in worker process."""
    global _worker_models
    
    worker_device = torch.device('cpu')
    
    vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(worker_device)
    rnn = RNN_MDN(latent_dim, 3, hidden_size, 5, 256, 1).to(worker_device)
    controller = Controller(input_features=latent_dim + hidden_size, actions_dims=3).to(worker_device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=worker_device))
    rnn.load_state_dict(torch.load(rnn_path, map_location=worker_device))
    
    vae.eval()
    rnn.eval()
    
    _worker_models = {
        'vae': vae,
        'rnn': rnn,
        'controller': controller,
        'device': worker_device,
        'envs_per_worker': envs_per_worker,
    }


def evaluate_candidate_worker(args):
    """Evaluate a single candidate in worker process."""
    params, max_steps = args
    global _worker_models
    
    vae = _worker_models['vae']
    rnn = _worker_models['rnn']
    controller = _worker_models['controller']
    device = _worker_models['device']
    envs_per_worker = _worker_models['envs_per_worker']
    
    # Set controller params
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size
    
    # Create fresh env for this evaluation
    vec_env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CarRacing-v3") for _ in range(envs_per_worker)
    ])
    
    try:
        with torch.no_grad():
            obs, _ = vec_env.reset()
            h = rnn.get_initial_hidden(device, batch_size=envs_per_worker)
            
            total_rewards = np.zeros(envs_per_worker)
            dones = np.zeros(envs_per_worker, dtype=bool)
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
            
            return float(np.mean(total_rewards))
    finally:
        vec_env.close()
        del vec_env
        gc.collect()


def train_controller_cma_parallel(
    vae_path,
    rnn_path,
    latent_dim=32,
    hidden_size=256,
    max_generations=100,
    population_size=32,
    sigma_init=0.5,
    eval_episodes=8,
    max_steps=1000,
    num_workers=2,
):
    """
    Train controller using CMA-ES with moderate parallelization.
    """
    # Calculate envs per worker
    envs_per_worker = eval_episodes  # Each worker evaluates all episodes for one candidate
    
    print(f"Starting training with {num_workers} workers")
    print(f"Each worker runs {envs_per_worker} parallel envs per candidate")
    
    # Initialize controller to get param count
    controller = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
    x0 = torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()
    num_params = len(x0)
    
    print(f"Training controller with {num_params} parameters")
    print(f"Population: {population_size}, Eval episodes: {eval_episodes}")
    
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
    
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(vae_path, rnn_path, latent_dim, hidden_size, envs_per_worker)
    ) as executor:
        
        try:
            while not es.stop():
                candidates = es.ask()
                
                # Prepare args
                args_list = [(c.astype(np.float32), max_steps) for c in candidates]
                
                # Evaluate in parallel
                rewards = list(tqdm(
                    executor.map(evaluate_candidate_worker, args_list),
                    total=len(candidates),
                    desc=f"Gen {generation}",
                    leave=False
                ))
                
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
                    params_tensor = torch.from_numpy(best_params).float()
                    idx = 0
                    for p in controller_save.parameters():
                        size = p.numel()
                        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
                        idx += size
                    torch.save(controller_save.state_dict(), "controller_cma_best.pth")
                
                print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards):7.1f} | "
                      f"Max: {np.max(rewards):7.1f} | Best ever: {best_reward:7.1f} | "
                      f"Sigma: {es.sigma:.4f}")
                
                if (generation + 1) % 10 == 0:
                    controller_save = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
                    params_tensor = torch.from_numpy(best_params).float()
                    idx = 0
                    for p in controller_save.parameters():
                        size = p.numel()
                        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
                        idx += size
                    torch.save(controller_save.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
                
                generation += 1
                
        except KeyboardInterrupt:
            print("\nTraining interrupted")
    
    return best_reward, best_params


def main():
    parser = argparse.ArgumentParser(description="Train controller with CMA-ES (Moderate Parallelism)")
    parser.add_argument("--vae", default="vae_weights_epoch_07.pth", help="VAE weights path")
    parser.add_argument("--rnn", default="weights_rnn/RNN_weights_epoch_02.pth", help="RNN weights path")
    parser.add_argument("--generations", type=int, default=100, help="Max generations")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial sigma")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Episodes per evaluation")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel workers")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CMA-ES Controller Training (Moderate Parallelism)")
    print("=" * 60)
    print(f"VAE: {args.vae}")
    print(f"RNN: {args.rnn}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num workers: {args.num_workers}")
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
    
    # Save final
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