# dream_visualization.py

import torch
import numpy as np
import os
from PIL import Image
import gymnasium as gym
from VAE import VAE
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_from_mdn(mu, sigma, pi):
    """
    Sample from Mixture Density Network output.
    
    Args:
        mu: [1, 1, num_gaussians, latent_dim]
        sigma: [1, 1, num_gaussians, latent_dim] (or var, depending on your implementation)
        pi: [1, 1, num_gaussians, latent_dim]
    
    Returns:
        z_next: [1, latent_dim]
    """
    # Remove batch and sequence dimensions
    mu = mu.squeeze(0).squeeze(0)       # [num_gaussians, latent_dim]
    sigma = sigma.squeeze(0).squeeze(0) # [num_gaussians, latent_dim]
    pi = pi.squeeze(0).squeeze(0)       # [num_gaussians, latent_dim]
    
    # Sample which Gaussian to use for each latent dimension
    # pi is [num_gaussians, latent_dim], need to sample per dimension
    pi_transposed = pi.T  # [latent_dim, num_gaussians]
    
    # Sample indices for each dimension
    indices = torch.multinomial(pi_transposed, 1).squeeze(-1)  # [latent_dim]
    
    # Gather selected mu and sigma
    latent_dim = mu.shape[1]
    mu_selected = mu[indices, torch.arange(latent_dim, device=mu.device)]       # [latent_dim]
    sigma_selected = sigma[indices, torch.arange(latent_dim, device=sigma.device)]  # [latent_dim]
    
    # Sample from selected Gaussians
    z_next = mu_selected + sigma_selected * torch.randn_like(mu_selected)
    
    return z_next.unsqueeze(0)  # [1, latent_dim]


def dream_rollout(
    vae, 
    rnn, 
    initial_obs, 
    action_space,
    num_steps=500,
    temperature=1.0,
    save_dir="dream_frames"
):
    """
    Generate a dream sequence using the world model.
    
    Args:
        vae: Trained VAE model
        rnn: Trained MDN-RNN model
        initial_obs: Initial observation from real environment [H, W, C] numpy array
        action_space: Gym action space for sampling random actions
        num_steps: Number of dream steps to generate
        temperature: Controls randomness in MDN sampling (higher = more random)
        save_dir: Directory to save frames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vae.eval()
    rnn.eval()
    
    frames = []
    
    with torch.no_grad():
        # Encode initial observation
        obs_tensor = torch.from_numpy(initial_obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Get initial latent
        mu, logvar = vae.encode(obs_tensor)
        z = vae.reparamterize(mu, logvar)  # [1, latent_dim]
        
        # Initialize RNN hidden state
        h = rnn.get_initial_hidden(device, batch_size=1)
        
        # Save initial real frame
        frames.append(initial_obs.copy())
        
        for step in range(num_steps):
            # Sample random action
            action = action_space.sample()
            a = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)  # [1, action_dim]
            
            # Get next z prediction from RNN
            (mu_next, var_next, pi_next), h = rnn.forward(z, h, a)
            
            # Sample next z from the mixture of Gaussians
            # Note: Check if your RNN outputs variance or sigma
            # If variance: sigma = torch.sqrt(var_next)
            # If already sigma: use directly
            sigma_next = torch.sqrt(var_next)  # Adjust if your model outputs sigma directly
            
            # Apply temperature
            sigma_next = sigma_next * temperature
            
            z = sample_from_mdn(mu_next, sigma_next, pi_next)
            
            # Decode z to image
            # Need to get the decoder_start_channels from VAE
            # This depends on your VAE implementation
            reconstructed = vae.decode(z, 128)  # Adjust 128 based on your VAE config
            
            # Convert to image
            img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, -1, 1)  # Tanh output is [-1, 1]
            img = ((img + 1) / 2 * 255).astype(np.uint8)  # Convert to [0, 255]
            
            frames.append(img)
            
            if step % 50 == 0:
                print(f"Dream step {step}/{num_steps}")
    
    # Save frames
    print(f"Saving {len(frames)} frames to {save_dir}/")
    for idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(save_dir, f"dream_{idx:04d}.png"))
    
    print(f"Done! Frames saved to {save_dir}/")
    return frames


def create_comparison_grid(real_frames, dream_frames, save_path="comparison.png"):
    """
    Create a side-by-side comparison of real vs dream frames.
    """
    import matplotlib.pyplot as plt
    
    n_frames = min(10, len(real_frames), len(dream_frames))
    fig, axes = plt.subplots(2, n_frames, figsize=(2*n_frames, 4))
    
    for i in range(n_frames):
        axes[0, i].imshow(real_frames[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Real {i}')
        
        axes[1, i].imshow(dream_frames[i])
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Dream {i}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Comparison saved to {save_path}")


def main():
    # Initialize environment (just to get initial observation and action space)
    env = gym.make("CarRacing-v3")
    
    # Initialize models
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(
        input_size=32,      # latent_dim
        action_dim=3,       # CarRacing action space
        hidden_size=35,     # Match your training config
        num_gaussians=5,
        hidden_layer=256,
        num_layers=1
    ).to(device)
    
    # Load trained weights
    vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))
    
    print("Models loaded!")
    
    # Get initial observation from real environment
    obs, _ = env.reset()
    
    # Run dream rollout
    dream_frames = dream_rollout(
        vae=vae,
        rnn=rnn,
        initial_obs=obs,
        action_space=env.action_space,
        num_steps=200,
        temperature=1.0,
        save_dir="dream_frames"
    )
    
    # Also collect some real frames for comparison
    print("Collecting real frames for comparison...")
    real_frames = [obs.copy()]
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        real_frames.append(obs.copy())
        if done:
            break
    
    # Create comparison
    create_comparison_grid(real_frames, dream_frames[:11], "real_vs_dream.png")
    
    env.close()


if __name__ == "__main__":
    main()