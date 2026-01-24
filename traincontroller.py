# train_controller_cma_vae_only.py

"""
Controller training using only VAE latent representations (no RNN).

This is a simpler baseline to verify:
1. The VAE is learning useful representations
2. The controller architecture works
3. CMA-ES optimization is functioning correctly

If this works, then we know any issues with the full model are likely
in the RNN integration, not the VAE or controller.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cma
from VAE import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ControllerVAEOnly(nn.Module):
    """
    Simple controller that maps VAE latent z directly to actions.
    
    Input: z (latent vector from VAE)
    Output: action (steering, gas, brake)
    """
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.net(z)


def evaluate_controller(vae, controller, env, max_steps=1000, render=False):
    """
    Evaluate controller using only VAE latents (no RNN).
    
    Args:
        vae: Trained VAE model
        controller: Controller that maps z -> action
        env: Gym environment
        max_steps: Maximum steps per episode
        render: Whether to render (for debugging)
    
    Returns:
        total_reward: Cumulative reward for the episode
    """
    vae.eval()
    controller.eval()
    
    with torch.no_grad():
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Encode observation to latent
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            # Use mean (mu) for deterministic behavior during evaluation
            # Alternatively, could sample: z = vae.reparamterize(mu, logvar)
            z = mu
            
            # Get action from controller
            action = controller(z)  # [1, 3], already in [-1, 1] due to tanh
            
            # Convert to numpy and scale to CarRacing action space
            a = action.squeeze(0).cpu().numpy()
            # steering stays in [-1, 1]
            a[1] = (a[1] + 1) / 2  # gas: [-1, 1] -> [0, 1]
            a[2] = (a[2] + 1) / 2  # brake: [-1, 1] -> [0, 1]
            a = a.astype(np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(a)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return total_reward


def get_controller_params(controller):
    """Flatten all controller parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()


def set_controller_params(controller, params):
    """Set controller parameters from a flattened vector."""
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx + size].view(p.shape))
        idx += size


def train_controller_cma(
    vae,
    controller,
    env,
    max_generations=100,
    population_size=16,
    sigma_init=0.5,
    eval_episodes=3,
    save_every=10,
):
    """
    Train controller using CMA-ES.
    
    Args:
        vae: Trained VAE (frozen)
        controller: Controller to optimize
        env: Gym environment
        max_generations: Maximum CMA-ES generations
        population_size: Number of candidates per generation
        sigma_init: Initial step size
        eval_episodes: Episodes to average for fitness
        save_every: Save checkpoint every N generations
    
    Returns:
        best_reward: Best reward achieved
    """
    vae.eval()
    
    # Get initial parameters
    x0 = get_controller_params(controller)
    num_params = len(x0)
    print(f"Training controller with {num_params} parameters using CMA-ES")
    print(f"Population size: {population_size}, Sigma: {sigma_init}")
    
    # CMA-ES options
    opts = {
        'popsize': population_size,
        'maxiter': max_generations,
        'CMA_diagonal': True,  # Faster for many parameters
        'verb_disp': 1,
        'verb_log': 0,
        'tolfun': 1e-11,  # Don't stop early on function tolerance
        'tolx': 1e-11,    # Don't stop early on parameter tolerance
    }
    
    # Initialize CMA-ES (minimizes, so we negate rewards)
    es = cma.CMAEvolutionStrategy(x0, sigma_init, opts)
    
    best_reward = -float('inf')
    best_params = x0.copy()
    generation = 0
    
    reward_history = []
    
    while not es.stop():
        # Get candidate solutions
        candidates = es.ask()
        
        # Evaluate each candidate
        fitnesses = []
        rewards_for_logging = []
        
        for i, candidate in enumerate(candidates):
            set_controller_params(controller, candidate)
            
            # Evaluate over multiple episodes for more stable fitness
            ep_rewards = []
            for ep in range(eval_episodes):
                r = evaluate_controller(vae, controller, env)
                ep_rewards.append(r)
            
            avg_reward = np.mean(ep_rewards)
            rewards_for_logging.append(avg_reward)
            
            # CMA-ES minimizes, so negate the reward
            fitnesses.append(-avg_reward)
        
        # Update CMA-ES
        es.tell(candidates, fitnesses)
        
        # Track best
        gen_best_idx = np.argmin(fitnesses)
        gen_best_reward = rewards_for_logging[gen_best_idx]
        gen_mean_reward = np.mean(rewards_for_logging)
        gen_max_reward = np.max(rewards_for_logging)
        
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_params = candidates[gen_best_idx].copy()
            # Save new best
            set_controller_params(controller, best_params)
            torch.save(controller.state_dict(), "controller_vae_only_best.pth")
        
        reward_history.append({
            'generation': generation,
            'mean': gen_mean_reward,
            'max': gen_max_reward,
            'best_ever': best_reward,
        })
        
        print(f"Gen {generation + 1:3d} | "
              f"Mean: {gen_mean_reward:7.1f} | "
              f"Max: {gen_max_reward:7.1f} | "
              f"Best ever: {best_reward:7.1f} | "
              f"Sigma: {es.sigma:.4f}")
        
        # Save checkpoint
        if (generation + 1) % save_every == 0:
            set_controller_params(controller, best_params)
            torch.save(controller.state_dict(), f"controller_vae_only_gen_{generation + 1:03d}.pth")
            np.save(f"reward_history_gen_{generation + 1:03d}.npy", reward_history)
        
        generation += 1
    
    print(f"\nCMA-ES stopped: {es.stop()}")
    
    # Restore best parameters
    set_controller_params(controller, best_params)
    
    return best_reward, reward_history


def random_baseline(env, num_episodes=10, max_steps=1000):
    """Evaluate random actions as a baseline."""
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train controller with VAE only (no RNN)")
    parser.add_argument("--vae_weights", type=str, default="vae_weights_epoch_04.pth",
                        help="Path to VAE weights")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="VAE latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Controller hidden layer dimension")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of CMA-ES generations")
    parser.add_argument("--population", type=int, default=16,
                        help="Population size per generation")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Initial CMA-ES step size")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Episodes to average for fitness evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Render final evaluation")
    args = parser.parse_args()
    
    # Create environment
    env = gym.make("CarRacing-v3", continuous=True)
    
    print("=" * 60)
    print("Controller Training (VAE Only - No RNN)")
    print("=" * 60)
    
    # Load VAE
    print(f"\nLoading VAE from {args.vae_weights}...")
    vae = VAE(
        in_channels=3,
        out_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=[32, 64, 128, 256],
    ).to(device)
    vae.load_state_dict(torch.load(args.vae_weights, map_location=device))
    vae.eval()
    print("VAE loaded!")
    
    # Initialize controller
    controller = ControllerVAEOnly(
        latent_dim=args.latent_dim,
        action_dim=3,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller parameters: {num_params}")
    
    # Random baseline
    print("\n" + "=" * 60)
    print("Random Action Baseline")
    print("=" * 60)
    rand_mean, rand_std = random_baseline(env, num_episodes=5)
    print(f"Random policy: {rand_mean:.1f} ± {rand_std:.1f}")
    
    # Untrained controller baseline
    print("\n" + "=" * 60)
    print("Untrained Controller Baseline")
    print("=" * 60)
    untrained_rewards = [evaluate_controller(vae, controller, env) for _ in range(5)]
    print(f"Untrained controller: {np.mean(untrained_rewards):.1f} ± {np.std(untrained_rewards):.1f}")
    
    # Train with CMA-ES
    print("\n" + "=" * 60)
    print("Training with CMA-ES")
    print("=" * 60)
    best_reward, history = train_controller_cma(
        vae=vae,
        controller=controller,
        env=env,
        max_generations=args.generations,
        population_size=args.population,
        sigma_init=args.sigma,
        eval_episodes=args.eval_episodes,
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    # Load best weights
    controller.load_state_dict(torch.load("controller_vae_only_best.pth", map_location=device))
    
    final_rewards = [evaluate_controller(vae, controller, env) for _ in range(10)]
    print(f"Trained controller: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"Best episode: {np.max(final_rewards):.1f}")
    print(f"Worst episode: {np.min(final_rewards):.1f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Random baseline:      {rand_mean:7.1f} ± {rand_std:.1f}")
    print(f"Untrained controller: {np.mean(untrained_rewards):7.1f} ± {np.std(untrained_rewards):.1f}")
    print(f"Trained controller:   {np.mean(final_rewards):7.1f} ± {np.std(final_rewards):.1f}")
    print(f"Improvement over random: {np.mean(final_rewards) - rand_mean:+.1f}")
    
    # Save final results
    np.save("reward_history_final.npy", history)
    torch.save(controller.state_dict(), "controller_vae_only_final.pth")
    print("\nSaved: controller_vae_only_final.pth, reward_history_final.npy")
    
    # Optional: render a few episodes
    if args.render:
        print("\n" + "=" * 60)
        print("Rendering trained controller...")
        print("=" * 60)
        env_render = gym.make("CarRacing-v3", continuous=True, render_mode="human")
        for ep in range(3):
            reward = evaluate_controller(vae, controller, env_render, render=True)
            print(f"Episode {ep + 1}: {reward:.1f}")
        env_render.close()
    
    env.close()


if __name__ == "__main__":
    main()