# train_controller_cma_efficient.py
"""
CMA-ES training with optional parallel workers.
Uses vectorized environments for efficiency.
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
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global models for worker processes
_worker_models = None


def init_worker(vae_path, rnn_path, latent_dim, hidden_size, eval_episodes):
    """Initialize models in worker process."""
    global _worker_models
    import os
    
    worker_id = os.getpid()
    
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
        'eval_episodes': eval_episodes,
        'worker_id': worker_id,
    }
    
    print(f"[Worker {worker_id}] Ready", flush=True)


def make_env():
    """Factory function for creating environments."""
    return gym.make("CarRacing-v3")


def evaluate_candidate_worker(args):
    """Evaluate a single candidate in worker process using multiple envs stepped together."""
    params, max_steps = args
    global _worker_models
    
    worker_id = _worker_models.get('worker_id', 'unknown')
    
    vae = _worker_models['vae']
    rnn = _worker_models['rnn']
    controller = _worker_models['controller']
    worker_device = _worker_models['device']
    eval_episodes = _worker_models['eval_episodes']
    
    print(f"[Worker {worker_id}] Setting controller params...", flush=True)
    
    # Set controller params
    params_tensor = torch.from_numpy(params).float().to(worker_device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size
    
    print(f"[Worker {worker_id}] Creating {eval_episodes} individual envs...", flush=True)
    
    # Create individual environments (no vectorization wrapper)
    envs = [gym.make("CarRacing-v3", render_mode=None) for _ in range(eval_episodes)]
    
    print(f"[Worker {worker_id}] Envs created, resetting...", flush=True)
    
    try:
        # Reset all envs
        observations = []
        for env in envs:
            obs, _ = env.reset()
            observations.append(obs)
        obs = np.stack(observations, axis=0)  # [num_envs, H, W, C]
        
        print(f"[Worker {worker_id}] Reset done, obs shape: {obs.shape}", flush=True)
        
        h = rnn.get_initial_hidden(worker_device, batch_size=eval_episodes)
        print(f"[Worker {worker_id}] Hidden state initialized, starting rollout...", flush=True)
        
        total_rewards = np.zeros(eval_episodes)
        dones = np.zeros(eval_episodes, dtype=bool)
        steps = 0
        
        with torch.no_grad():
            while not np.all(dones) and steps < max_steps:
                if steps % 100 == 0:
                    print(f"[Worker {worker_id}] Step {steps}, {np.sum(~dones)} envs still active...", flush=True)
                
                # Process observations
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(worker_device)
                
                # VAE encode
                mu, logvar = vae.encode(obs_tensor)
                z = vae.reparamterize(mu, logvar)
                
                # Controller
                h_for_controller = h[0][-1]
                controller_input = torch.cat([z, h_for_controller], dim=1)
                
                actions = controller(controller_input)
                actions = torch.tanh(actions)
                
                # Scale actions
                a = actions.cpu().numpy()
                a[:, 1] = (a[:, 1] + 1) / 2
                a[:, 2] = (a[:, 2] + 1) / 2
                a = a.astype(np.float32)
                
                # Step all envs individually
                new_observations = []
                rewards = np.zeros(eval_episodes)
                done_now = np.zeros(eval_episodes, dtype=bool)
                
                for i, env in enumerate(envs):
                    if not dones[i]:
                        new_obs, reward, terminated, truncated, _ = env.step(a[i])
                        new_observations.append(new_obs)
                        rewards[i] = reward
                        done_now[i] = terminated or truncated
                    else:
                        new_observations.append(obs[i])  # Keep old obs for done envs
                
                obs = np.stack(new_observations, axis=0)
                
                # Accumulate rewards for non-done envs
                total_rewards += rewards * (~dones)
                dones = dones | done_now
                
                # Update RNN hidden state
                a_tensor = torch.from_numpy(a).float().to(worker_device)
                (_, _, _), h = rnn.forward(z, h, a_tensor)
                
                steps += 1
        
        avg_reward = float(np.mean(total_rewards))
        print(f"[Worker {worker_id}] Done: {avg_reward:.1f} (steps={steps})", flush=True)
        return avg_reward
        
    finally:
        for env in envs:
            env.close()
        gc.collect()


def evaluate_candidate_worker_OLD(args):
    """Evaluate a single candidate in worker process - vectorized version (may deadlock)."""
    params, max_steps = args
    global _worker_models
    
    vae = _worker_models['vae']
    rnn = _worker_models['rnn']
    controller = _worker_models['controller']
    worker_device = _worker_models['device']
    eval_episodes = _worker_models['eval_episodes']
    
    # Set controller params
    params_tensor = torch.from_numpy(params).float().to(worker_device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size
    
    # Create vectorized env
    vec_env = gym.vector.SyncVectorEnv([
        make_env for _ in range(eval_episodes)
    ])
    
def evaluate_controller_vectorized(vae, rnn, controller, num_envs, max_steps=1000):
    """
    Evaluate controller with a fresh vectorized environment.
    Creates and destroys env each call to manage memory.
    """
    print(f"  Creating {num_envs} vectorized envs...", flush=True)
    
    # Create env
    vec_env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CarRacing-v3", render_mode=None) for _ in range(num_envs)
    ])
    
    print(f"  Envs created, resetting...", flush=True)
    
    try:
        vae.eval()
        rnn.eval()
        controller.eval()
        
        with torch.no_grad():
            obs, _ = vec_env.reset()
            print(f"  Reset done, starting rollout...", flush=True)
            h = rnn.get_initial_hidden(device, batch_size=num_envs)
            
            total_rewards = np.zeros(num_envs)
            dones = np.zeros(num_envs, dtype=bool)
            steps = 0
            
            while not np.all(dones) and steps < max_steps:
                if steps % 100 == 0:
                    print(f"  Step {steps}, {np.sum(~dones)} envs active...", flush=True)
                
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
    finally:
        vec_env.close()
        del vec_env
        gc.collect()


def get_controller_params(controller):
    return torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()


def set_controller_params(controller, params, target_device=None):
    if target_device is None:
        target_device = device
    params_tensor = torch.from_numpy(params).float().to(target_device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size


def train_controller_cma(
    vae,
    rnn,
    controller,
    max_generations=100,
    population_size=32,
    sigma_init=0.5,
    eval_episodes=8,
    max_steps=1000,
    candidates_parallel=4,
):
    """
    Train controller using CMA-ES with batched candidate evaluation.
    Evaluates multiple candidates in parallel using a large vectorized environment.
    """
    vae.eval()
    rnn.eval()
    
    x0 = get_controller_params(controller)
    num_params = len(x0)
    
    total_envs = candidates_parallel * eval_episodes
    print(f"Training controller with {num_params} parameters")
    print(f"Population: {population_size}, Eval episodes: {eval_episodes}")
    print(f"Candidates in parallel: {candidates_parallel}, Total envs: {total_envs}")
    
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
    
    # Create multiple controllers for parallel evaluation
    # Get input/output dimensions from the original controller
    input_dim = controller.fc1.in_features
    output_dim = controller.fc1.out_features  # Controller is just a single linear layer
    
    controllers = [Controller(input_features=input_dim, 
                              actions_dims=output_dim).to(device) 
                   for _ in range(candidates_parallel)]
    
    try:
        while not es.stop():
            candidates = es.ask()
            rewards = []
            
            # Process candidates in batches
            num_batches = (len(candidates) + candidates_parallel - 1) // candidates_parallel
            
            pbar = tqdm(total=len(candidates), desc=f"Gen {generation}", leave=False)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * candidates_parallel
                end_idx = min(start_idx + candidates_parallel, len(candidates))
                batch_candidates = candidates[start_idx:end_idx]
                actual_batch_size = len(batch_candidates)
                
                # Set params for each controller in batch
                for i, candidate in enumerate(batch_candidates):
                    set_controller_params(controllers[i], candidate)
                
                # Evaluate batch
                batch_rewards = evaluate_candidates_batched(
                    vae, rnn, controllers[:actual_batch_size],
                    eval_episodes=eval_episodes,
                    max_steps=max_steps
                )
                
                rewards.extend(batch_rewards)
                pbar.update(actual_batch_size)
                pbar.set_postfix({'last_reward': f'{batch_rewards[-1]:.1f}'})
            
            pbar.close()
            
            fitnesses = [-r for r in rewards]
            es.tell(candidates, fitnesses)
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best_reward = rewards[gen_best_idx]
            
            if gen_best_reward > best_reward:
                best_reward = gen_best_reward
                best_params = candidates[gen_best_idx].copy()
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), "controller_cma_best.pth")
            
            print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards):7.1f} | "
                  f"Max: {np.max(rewards):7.1f} | Best ever: {best_reward:7.1f} | "
                  f"Sigma: {es.sigma:.4f}")
            
            if (generation + 1) % 10 == 0:
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
            
            generation += 1
            gc.collect()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    set_controller_params(controller, best_params)
    return best_reward


def evaluate_candidates_batched(vae, rnn, controllers, eval_episodes, max_steps):
    """
    Evaluate multiple candidates simultaneously using batched environments.
    
    Args:
        vae: VAE model
        rnn: RNN model  
        controllers: List of controllers (one per candidate)
        eval_episodes: Episodes per candidate
        max_steps: Max steps per episode
    
    Returns:
        List of average rewards, one per candidate
    """
    num_candidates = len(controllers)
    total_envs = num_candidates * eval_episodes
    
    # Create vectorized environment for all candidates
    vec_env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CarRacing-v3", render_mode=None) for _ in range(total_envs)
    ])
    
    try:
        vae.eval()
        rnn.eval()
        for c in controllers:
            c.eval()
        
        with torch.no_grad():
            obs, _ = vec_env.reset()
            
            # Initialize hidden states for all envs
            h = rnn.get_initial_hidden(device, batch_size=total_envs)
            
            total_rewards = np.zeros(total_envs)
            dones = np.zeros(total_envs, dtype=bool)
            steps = 0
            
            while not np.all(dones) and steps < max_steps:
                # Process all observations through VAE
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(device)
                
                mu, logvar = vae.encode(obs_tensor)
                z = vae.reparamterize(mu, logvar)
                
                # Get hidden state for controller
                h_for_controller = h[0][-1]  # [total_envs, hidden_size]
                controller_input = torch.cat([z, h_for_controller], dim=1)
                
                # Apply each controller to its corresponding envs
                actions_list = []
                for i, ctrl in enumerate(controllers):
                    start_env = i * eval_episodes
                    end_env = start_env + eval_episodes
                    
                    ctrl_input = controller_input[start_env:end_env]
                    action = ctrl(ctrl_input)
                    action = torch.tanh(action)
                    actions_list.append(action)
                
                actions = torch.cat(actions_list, dim=0)
                
                # Scale actions
                a = actions.cpu().numpy()
                a[:, 1] = (a[:, 1] + 1) / 2
                a[:, 2] = (a[:, 2] + 1) / 2
                a = a.astype(np.float32)
                
                # Step all envs
                obs, rewards, terminated, truncated, _ = vec_env.step(a)
                done_now = terminated | truncated
                
                # Accumulate rewards
                total_rewards += rewards * (~dones)
                dones = dones | done_now
                
                # Update RNN hidden state
                a_tensor = torch.from_numpy(a).float().to(device)
                (_, _, _), h = rnn.forward(z, h, a_tensor)
                
                steps += 1
        
        # Compute average reward per candidate
        candidate_rewards = []
        for i in range(num_candidates):
            start_env = i * eval_episodes
            end_env = start_env + eval_episodes
            avg_reward = np.mean(total_rewards[start_env:end_env])
            candidate_rewards.append(avg_reward)
        
        return candidate_rewards
        
    finally:
        vec_env.close()
        del vec_env
        gc.collect()


def train_controller_cma_OLD(
    vae,
    rnn,
    controller,
    max_generations=100,
    population_size=32,
    sigma_init=0.5,
    eval_episodes=8,
    max_steps=1000,
):
    """
    Train controller using CMA-ES with memory-efficient evaluation (no workers).
    OLD VERSION - evaluates one candidate at a time.
    """
    vae.eval()
    rnn.eval()
    
    x0 = get_controller_params(controller)
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
    
    try:
        while not es.stop():
            candidates = es.ask()
            
            rewards = []
            
            pbar = tqdm(candidates, desc=f"Gen {generation}", leave=False)
            for candidate in pbar:
                set_controller_params(controller, candidate)
                
                # Evaluate with vectorized env
                episode_rewards = evaluate_controller_vectorized(
                    vae, rnn, controller, 
                    num_envs=eval_episodes,
                    max_steps=max_steps
                )
                
                avg_reward = np.mean(episode_rewards)
                rewards.append(avg_reward)
                
                pbar.set_postfix({'reward': f'{avg_reward:.1f}'})
            
            fitnesses = [-r for r in rewards]
            es.tell(candidates, fitnesses)
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best_reward = rewards[gen_best_idx]
            
            if gen_best_reward > best_reward:
                best_reward = gen_best_reward
                best_params = candidates[gen_best_idx].copy()
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), "controller_cma_best.pth")
            
            print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards):7.1f} | "
                  f"Max: {np.max(rewards):7.1f} | Best ever: {best_reward:7.1f} | "
                  f"Sigma: {es.sigma:.4f}")
            
            if (generation + 1) % 10 == 0:
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
            
            generation += 1
            gc.collect()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    set_controller_params(controller, best_params)
    return best_reward


def train_controller_cma_parallel(
    vae_path,
    rnn_path,
    latent_dim,
    hidden_size,
    max_generations=100,
    population_size=64,
    sigma_init=0.5,
    eval_episodes=16,
    max_steps=1000,
    num_workers=8,
):
    """
    Train controller using CMA-ES with parallel workers.
    """
    print(f"Training with {num_workers} parallel workers")
    print(f"Each worker runs {eval_episodes} vectorized envs")
    
    # Initialize controller to get params
    controller = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
    x0 = get_controller_params(controller)
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
    
    print("Creating process pool...", flush=True)
    print(f"Initializer args: vae={vae_path}, rnn={rnn_path}", flush=True)
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(vae_path, rnn_path, latent_dim, hidden_size, eval_episodes)
    ) as executor:
        
        print("Process pool created, starting training loop...", flush=True)
        
        try:
            while not es.stop():
                candidates = es.ask()
                
                # Prepare args
                args_list = [(c.astype(np.float32), max_steps) for c in candidates]
                
                # Evaluate all candidates in parallel
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
                    set_controller_params(controller_save, best_params, target_device='cpu')
                    torch.save(controller_save.state_dict(), "controller_cma_best.pth")
                
                print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards):7.1f} | "
                      f"Max: {np.max(rewards):7.1f} | Best ever: {best_reward:7.1f} | "
                      f"Sigma: {es.sigma:.4f}")
                
                if (generation + 1) % 10 == 0:
                    controller_save = Controller(input_features=latent_dim + hidden_size, actions_dims=3)
                    set_controller_params(controller_save, best_params, target_device='cpu')
                    torch.save(controller_save.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
                
                generation += 1
                
        except KeyboardInterrupt:
            print("\nTraining interrupted")
    
    return best_reward, best_params


def main():
    parser = argparse.ArgumentParser(description="Train controller with CMA-ES")
    parser.add_argument("--vae", default="vae_weights_epoch_07.pth", help="VAE weights path")
    parser.add_argument("--rnn", default="weights_rnn/RNN_weights_epoch_02.pth", help="RNN weights path")
    parser.add_argument("--generations", type=int, default=100, help="Max generations")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial sigma")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Episodes per evaluation")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of parallel workers (0 = no parallelism)")
    parser.add_argument("--candidates-parallel", type=int, default=4, help="Number of candidates to evaluate in parallel (when num-workers=0)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CMA-ES Controller Training")
    print("=" * 60)
    print(f"VAE: {args.vae}")
    print(f"RNN: {args.rnn}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Max steps: {args.max_steps}")
    if args.num_workers > 0:
        print(f"Num workers: {args.num_workers}")
    else:
        print(f"Candidates in parallel: {args.candidates_parallel}")
        print(f"Total envs: {args.candidates_parallel * args.eval_episodes}")
    print(f"Device: {device}")
    print("=" * 60)
    
    latent_dim = 32
    hidden_size = 256
    
    if args.num_workers > 0:
        # Parallel training
        print(f"\nUsing {args.num_workers} parallel workers")
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
        set_controller_params(controller_final, best_params, target_device='cpu')
        torch.save(controller_final.state_dict(), "controller_cma_final.pth")
        
    else:
        # Batched training (multiple candidates in parallel)
        print("Loading models...")
        vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(device)
        rnn = RNN_MDN(latent_dim, 3, hidden_size, 5, 256, 1).to(device)
        
        vae.load_state_dict(torch.load(args.vae, map_location=device))
        rnn.load_state_dict(torch.load(args.rnn, map_location=device))
        
        controller = Controller(
            input_features=latent_dim + hidden_size,
            actions_dims=3,
        ).to(device)
        
        print("Models loaded!")
        
        # Baseline
        print("\n=== Baseline ===")
        baseline_rewards = evaluate_controller_vectorized(
            vae, rnn, controller, num_envs=4, max_steps=args.max_steps
        )
        print(f"Untrained: {np.mean(baseline_rewards):.1f} ± {np.std(baseline_rewards):.1f}")
        
        # Train
        print("\n=== Training ===")
        best_reward = train_controller_cma(
            vae, rnn, controller,
            max_generations=args.generations,
            population_size=args.population,
            sigma_init=args.sigma,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            candidates_parallel=args.candidates_parallel,
        )
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        final_rewards = evaluate_controller_vectorized(
            vae, rnn, controller, num_envs=8, max_steps=args.max_steps
        )
        print(f"Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
        
        torch.save(controller.state_dict(), "controller_cma_final.pth")
    
    print(f"\nTraining complete! Best reward: {best_reward:.1f}")
    print("Saved: controller_cma_final.pth, controller_cma_best.pth")


if __name__ == "__main__":
    main()