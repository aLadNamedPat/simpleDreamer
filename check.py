# train_controller_cma.py

import torch
import numpy as np
import gymnasium as gym
import cma
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_controller(vae, rnn, controller, env, max_steps=1000):
    """Evaluate controller in real environment. Returns total reward."""
    vae.eval()
    rnn.eval()
    controller.eval()
    
    with torch.no_grad():
        obs, _ = env.reset()
        h = rnn.get_initial_hidden(device, batch_size=1)
        total_reward = 0
        
        for step in range(max_steps):
            # Encode observation
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            # Get action from controller
            h_for_controller = h[0][-1]  # [1, hidden_size]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            action = controller(controller_input)
            action = torch.tanh(action)
            
            # Scale to CarRacing action space
            a = action.squeeze(0).cpu().numpy()
            a[1] = (a[1] + 1) / 2  # gas: [-1,1] -> [0,1]
            a[2] = (a[2] + 1) / 2  # brake: [-1,1] -> [0,1]
            a = a.astype(np.float32)
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(a)
            total_reward += reward
            
            # Update RNN hidden state
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            if done or truncated:
                break
        
        return total_reward


def get_controller_params(controller):
    """Flatten all controller parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in controller.parameters()]).cpu().numpy()


def set_controller_params(controller, params):
    """Set controller parameters from a flattened vector (numpy array)."""
    params_tensor = torch.from_numpy(params).float().to(device)
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params_tensor[idx:idx+size].view(p.shape))
        idx += size


def train_controller_cma(
    vae, 
    rnn, 
    controller, 
    env,
    max_generations=100,
    population_size=16,
    sigma_init=0.5,
    eval_episodes=1
):
    """
    Train controller using CMA-ES.
    """
    vae.eval()
    rnn.eval()
    
    # Get initial parameters
    x0 = get_controller_params(controller)
    num_params = len(x0)
    print(f"Training controller with {num_params} parameters using CMA-ES")
    
    # CMA-ES options
    opts = {
        'popsize': population_size,
        'maxiter': max_generations,
        'CMA_diagonal': True,  # Use diagonal covariance (faster for many params)
        'verb_disp': 1,        # Print every generation
        'verb_log': 0,         # No file logging
    }
    
    # Initialize CMA-ES
    # Note: CMA-ES minimizes, so we'll negate rewards
    es = cma.CMAEvolutionStrategy(x0, sigma_init, opts)
    
    best_reward = -float('inf')
    best_params = x0.copy()
    generation = 0
    
    while not es.stop():
        # Get candidate solutions
        candidates = es.ask()
        
        # Evaluate each candidate
        fitnesses = []
        rewards_for_logging = []
        
        for i, candidate in enumerate(candidates):
            set_controller_params(controller, candidate)
            
            # Evaluate over multiple episodes
            ep_rewards = []
            for _ in range(eval_episodes):
                r = evaluate_controller(vae, rnn, controller, env)
                ep_rewards.append(r)
            
            avg_reward = np.mean(ep_rewards)
            rewards_for_logging.append(avg_reward)
            
            # CMA-ES minimizes, so negate the reward
            fitnesses.append(-avg_reward)
            
            print(f"  Gen {generation} | Candidate {i+1}/{len(candidates)} | Reward: {avg_reward:.1f}")
        
        # Update CMA-ES
        es.tell(candidates, fitnesses)
        
        # Track best
        gen_best_idx = np.argmin(fitnesses)
        gen_best_reward = rewards_for_logging[gen_best_idx]
        
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_params = candidates[gen_best_idx].copy()
        
        print(f"Gen {generation+1:3d} | Mean: {np.mean(rewards_for_logging):7.1f} | "
              f"Max: {np.max(rewards_for_logging):7.1f} | Best ever: {best_reward:7.1f} | "
              f"Sigma: {es.sigma:.4f}")
        
        # Save checkpoint every 10 generations
        if (generation + 1) % 10 == 0:
            set_controller_params(controller, best_params)
            torch.save(controller.state_dict(), f"controller_cma_gen_{generation+1:03d}.pth")
        
        generation += 1
    
    # Print stop reason
    print(f"\nCMA-ES stopped: {es.stop()}")
    
    # Restore best parameters
    set_controller_params(controller, best_params)
    return best_reward


def main():
    env = gym.make("CarRacing-v3")
    
    # Load VAE and RNN
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_04.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_50.pth", map_location=device))
    
    # Initialize controller
    controller = Controller(
        input_features=32 + 35,
        actions_dims=3,
    ).to(device)
    
    print("Models loaded!")
    
    # Baseline
    print("\n=== Baseline ===")
    baseline = np.mean([evaluate_controller(vae, rnn, controller, env) for _ in range(3)])
    print(f"Untrained controller: {baseline:.1f}")
    
    # Train with CMA-ES
    print("\n=== Training Controller with CMA-ES ===")
    best_reward = train_controller_cma(
        vae, rnn, controller, env,
        max_generations=100,
        population_size=16,
        sigma_init=0.5,      # Initial step size (often larger than vanilla ES)
        eval_episodes=1
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_rewards = [evaluate_controller(vae, rnn, controller, env) for _ in range(10)]
    print(f"Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    
    # Save final controller
    torch.save(controller.state_dict(), "controller_cma_final.pth")
    print("Saved controller_cma_final.pth")
    
    env.close()


if __name__ == "__main__":
    main()