# controller_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_from_mdn(mu, var, pi):
    """Sample z from MDN output."""
    mu = mu.squeeze(0).squeeze(0)
    sigma = torch.sqrt(var).squeeze(0).squeeze(0)
    pi = pi.squeeze(0).squeeze(0)
    
    pi_t = pi.T
    indices = torch.multinomial(pi_t, 1).squeeze(-1)
    
    latent_dim = mu.shape[1]
    mu_sel = mu[indices, torch.arange(latent_dim, device=mu.device)]
    sigma_sel = sigma[indices, torch.arange(latent_dim, device=sigma.device)]
    
    z_next = mu_sel + sigma_sel * torch.randn_like(mu_sel)
    return z_next.unsqueeze(0)


def dream_rollout_with_controller(vae, rnn, controller, initial_z, max_steps=500, temperature=1.0):
    """
    Run controller inside the dream world.
    Returns total reward proxy (negative reconstruction uncertainty or similar).
    """
    z = initial_z
    h = rnn.get_initial_hidden(device, batch_size=1)
    
    # For CarRacing, reward proxy: how long we survive / stay on track
    # In dream, we don't have true reward, so we use a proxy
    total_steps = 0
    
    for step in range(max_steps):
        # Controller takes [z, h] and outputs action
        h_for_controller = h[0][-1]  # Last layer hidden state [1, hidden_size]
        controller_input = torch.cat([z, h_for_controller], dim=1)
        
        action = controller(controller_input)
        action = torch.tanh(action)  # Bound actions to [-1, 1]
        
        # Scale actions to CarRacing range
        # steering: [-1, 1], gas: [0, 1], brake: [0, 1]
        a = action.clone()
        a[:, 1] = (a[:, 1] + 1) / 2  # gas: [-1,1] -> [0,1]
        a[:, 2] = (a[:, 2] + 1) / 2  # brake: [-1,1] -> [0,1]
        
        # Step in dream world
        (mu_next, var_next, pi_next), h = rnn.forward(z, h, a)
        
        sigma_next = torch.sqrt(var_next) * temperature
        z = sample_from_mdn(mu_next, var_next, pi_next)
        
        total_steps += 1
    
    return total_steps  # Simple proxy: survived all steps


def evaluate_in_real_env(vae, rnn, controller, env, num_episodes=5, max_steps=1000):
    """Evaluate controller in real environment."""
    vae.eval()
    rnn.eval()
    controller.eval()
    
    total_rewards = []
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            h = rnn.get_initial_hidden(device, batch_size=1)
            episode_reward = 0
            
            for step in range(max_steps):
                # Encode observation
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                mu, logvar = vae.encode(obs_tensor)
                z = vae.reparamterize(mu, logvar)
                
                # Get action from controller
                h_for_controller = h[0][-1]
                controller_input = torch.cat([z, h_for_controller], dim=1)
                
                action = controller(controller_input)
                action = torch.tanh(action)
                
                # Scale to CarRacing
                a = action.squeeze(0).cpu().numpy()
                a[1] = (a[1] + 1) / 2  # gas
                a[2] = (a[2] + 1) / 2  # brake
                
                # Step environment
                obs, reward, done, truncated, _ = env.step(a)
                episode_reward += reward
                
                # Update RNN hidden state
                a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
                (_, _, _), h = rnn.forward(z, h, a_tensor)
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"Episode {ep+1}: {episode_reward:.1f}")
    
    return np.mean(total_rewards), np.std(total_rewards)


def main():
    # Load models
    env = gym.make("CarRacing-v3")
    
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))
    
    # Initialize controller
    # Input: z (32) + h (35) = 67
    controller = Controller(
        input_features=32 + 35,  # z_dim + hidden_size
        actions_dims=3,
        action_space=env.action_space
    ).to(device)
    
    print("Models loaded!")
    
    # First: Evaluate random controller as baseline
    print("\n=== Baseline (Untrained Controller) ===")
    mean_reward, std_reward = evaluate_in_real_env(vae, rnn, controller, env, num_episodes=5)
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    
    env.close()


if __name__ == "__main__":
    main()