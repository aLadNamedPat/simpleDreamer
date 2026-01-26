# train_controller_cma_vectorized.py

import torch
import numpy as np
import gymnasium as gym
import cma
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_controller_vectorized(vae, rnn, controller, vec_env, max_steps=1000):
    """
    Evaluate controller across multiple environments in parallel.
    
    Args:
        vae: VAE model
        rnn: RNN-MDN model
        controller: Controller model
        vec_env: Vectorized environment (gym.vector.SyncVectorEnv or AsyncVectorEnv)
        max_steps: Maximum steps per episode
    
    Returns:
        Array of total rewards for each environment
    """
    num_envs = vec_env.num_envs
    
    vae.eval()
    rnn.eval()
    controller.eval()
    
    with torch.no_grad():
        # Reset all environments
        obs, _ = vec_env.reset()
        
        # Initialize hidden states for all environments [num_layers, num_envs, hidden_size]
        h = rnn.get_initial_hidden(device, batch_size=num_envs)
        
        total_rewards = np.zeros(num_envs)
        dones = np.zeros(num_envs, dtype=bool)
        steps = 0
        
        while not np.all(dones) and steps < max_steps:
            # Encode observations: [num_envs, H, W, C] -> [num_envs, C, H, W]
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(device)
            
            # VAE encode
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)  # [num_envs, latent_dim]
            
            # Get action from controller
            h_for_controller = h[0][-1]  # [num_envs, hidden_size]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            actions = controller(controller_input)
            actions = torch.tanh(actions)
            
            # Scale to CarRacing action space
            a = actions.cpu().numpy()
            a[:, 1] = (a[:, 1] + 1) / 2  # gas: [-1,1] -> [0,1]
            a[:, 2] = (a[:, 2] + 1) / 2  # brake: [-1,1] -> [0,1]
            a = a.astype(np.float32)
            
            # Step all environments
            obs, rewards, terminated, truncated, _ = vec_env.step(a)
            done_now = terminated | truncated
            
            # Accumulate rewards only for non-done environments
            total_rewards += rewards * (~dones)
            dones = dones | done_now
            
            # Update RNN hidden state
            a_tensor = torch.from_numpy(a).float().to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            steps += 1
        
        return total_rewards


def get_controller_params(controller):
    """Flatten all controller parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()


def set_controller_params(controller, params):
    """Set controller parameters from a flattened vector."""
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size


def train_controller_cma_vectorized(
    vae,
    rnn,
    controller,
    env_name="CarRacing-v3",
    max_generations=100,
    population_size=64,
    sigma_init=0.5,
    eval_episodes=16,
    max_steps=1000,
):
    """
    Train controller using CMA-ES with vectorized environments.
    
    Args:
        vae: Trained VAE model
        rnn: Trained RNN-MDN model
        controller: Controller to train
        env_name: Gymnasium environment name
        max_generations: Maximum CMA-ES generations
        population_size: Number of candidates per generation
        sigma_init: Initial CMA-ES sigma
        eval_episodes: Number of episodes to average for fitness
        max_steps: Max steps per episode
    """
    vae.eval()
    rnn.eval()
    
    # Create vectorized environment for parallel evaluation
    vec_env = gym.vector.SyncVectorEnv([
        lambda: gym.make(env_name) for _ in range(eval_episodes)
    ])
    
    print(f"Created vectorized environment with {eval_episodes} parallel envs")
    
    # Get initial parameters
    x0 = get_controller_params(controller)
    num_params = len(x0)
    print(f"Training controller with {num_params} parameters using CMA-ES")
    print(f"Population: {population_size}, Eval episodes: {eval_episodes}")
    
    # CMA-ES options
    opts = {
        'popsize': population_size,
        'maxiter': max_generations,
        'CMA_diagonal': True,
        'verb_disp': 1,
        'verb_log': 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma_init, opts)
    
    best_reward = -float('inf')
    best_params = x0.copy()
    generation = 0
    
    try:
        while not es.stop():
            candidates = es.ask()
            
            fitnesses = []
            rewards_for_logging = []
            
            # Progress bar for candidates in this generation
            pbar = tqdm(enumerate(candidates), total=len(candidates), 
                       desc=f"Gen {generation}", leave=False)
            
            for i, candidate in pbar:
                set_controller_params(controller, candidate)
                
                # Evaluate across all episodes in parallel
                episode_rewards = evaluate_controller_vectorized(
                    vae, rnn, controller, vec_env, max_steps
                )
                
                avg_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                rewards_for_logging.append(avg_reward)
                
                # CMA-ES minimizes, so negate
                fitnesses.append(-avg_reward)
                
                pbar.set_postfix({'reward': f'{avg_reward:.1f}±{std_reward:.1f}'})
            
            es.tell(candidates, fitnesses)
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best_reward = rewards_for_logging[gen_best_idx]
            
            if gen_best_reward > best_reward:
                best_reward = gen_best_reward
                best_params = candidates[gen_best_idx].copy()
                # Save best immediately
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), "controller_cma_best.pth")
            
            print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards_for_logging):7.1f} | "
                  f"Max: {np.max(rewards_for_logging):7.1f} | Best ever: {best_reward:7.1f} | "
                  f"Sigma: {es.sigma:.4f}")
            
            # Save checkpoint every 10 generations
            if (generation + 1) % 10 == 0:
                set_controller_params(controller, best_params)
                torch.save(controller.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
            
            generation += 1
    
    finally:
        vec_env.close()
    
    print(f"\nCMA-ES stopped: {es.stop()}")
    
    # Restore best parameters
    set_controller_params(controller, best_params)
    return best_reward


def main():
    parser = argparse.ArgumentParser(description="Train controller with CMA-ES (Vectorized)")
    parser.add_argument("--vae", default="vae_weights_epoch_07.pth", help="VAE weights path")
    parser.add_argument("--rnn", default="weights_rnn/RNN_weights_epoch_02.pth", help="RNN weights path")
    parser.add_argument("--generations", type=int, default=100, help="Max generations")
    parser.add_argument("--population", type=int, default=64, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial sigma")
    parser.add_argument("--eval-episodes", type=int, default=16, help="Episodes per evaluation")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    args = parser.parse_args()
    
    # Model architecture parameters
    latent_dim = 32
    hidden_size = 256
    
    # Load models
    print("Loading models...")
    vae = VAE(3, 3, latent_dim, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(latent_dim, 3, hidden_size, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load(args.vae, map_location=device))
    rnn.load_state_dict(torch.load(args.rnn, map_location=device))
    
    # Initialize controller
    controller = Controller(
        input_features=latent_dim + hidden_size,
        actions_dims=3,
    ).to(device)
    
    print("Models loaded!")
    print(f"VAE: {args.vae}")
    print(f"RNN: {args.rnn}")
    print(f"Controller input size: {latent_dim + hidden_size}")
    print(f"Device: {device}")
    
    # Create single env for baseline
    single_env = gym.make("CarRacing-v3")
    
    # Quick baseline with vectorized env
    print("\n=== Baseline (3 episodes) ===")
    vec_env_baseline = gym.vector.SyncVectorEnv([
        lambda: gym.make("CarRacing-v3") for _ in range(3)
    ])
    baseline_rewards = evaluate_controller_vectorized(vae, rnn, controller, vec_env_baseline, args.max_steps)
    vec_env_baseline.close()
    print(f"Untrained controller: {np.mean(baseline_rewards):.1f} ± {np.std(baseline_rewards):.1f}")
    
    # Train
    print("\n=== Training Controller with CMA-ES (Vectorized) ===")
    best_reward = train_controller_cma_vectorized(
        vae, rnn, controller,
        env_name="CarRacing-v3",
        max_generations=args.generations,
        population_size=args.population,
        sigma_init=args.sigma,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )
    
    # Final evaluation
    print("\n=== Final Evaluation (16 episodes) ===")
    vec_env_final = gym.vector.SyncVectorEnv([
        lambda: gym.make("CarRacing-v3") for _ in range(16)
    ])
    final_rewards = evaluate_controller_vectorized(vae, rnn, controller, vec_env_final, args.max_steps)
    vec_env_final.close()
    print(f"Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"Min: {np.min(final_rewards):.1f}, Max: {np.max(final_rewards):.1f}")
    
    # Save final
    torch.save(controller.state_dict(), "controller_cma_final.pth")
    print("\nSaved controller_cma_final.pth")


if __name__ == "__main__":
    main()