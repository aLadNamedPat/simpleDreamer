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
    """Sample from Mixture Density Network output."""
    mu = mu.squeeze(0).squeeze(0)
    sigma = sigma.squeeze(0).squeeze(0)
    pi = pi.squeeze(0).squeeze(0)
    
    pi_transposed = pi.T
    indices = torch.multinomial(pi_transposed, 1).squeeze(-1)
    
    latent_dim = mu.shape[1]
    mu_selected = mu[indices, torch.arange(latent_dim, device=mu.device)]
    sigma_selected = sigma[indices, torch.arange(latent_dim, device=sigma.device)]
    
    z_next = mu_selected + sigma_selected * torch.randn_like(mu_selected)
    return z_next.unsqueeze(0)


def dream_rollout(vae, rnn, initial_obs, action_space, num_steps=500, temperature=1.0):
    """Generate a dream sequence using the world model."""
    vae.eval()
    rnn.eval()
    
    frames = []
    
    with torch.no_grad():
        obs_tensor = torch.from_numpy(initial_obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        mu, logvar = vae.encode(obs_tensor)
        z = vae.reparamterize(mu, logvar)
        
        h = rnn.get_initial_hidden(device, batch_size=1)
        
        for step in range(num_steps):
            # Decode current z to image
            reconstructed = vae.decode(z, 128)
            
            img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, -1, 1)
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            frames.append(img)
            
            # Sample random action
            action = action_space.sample()
            a = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
            
            # Predict next z
            (mu_next, var_next, pi_next), h = rnn.forward(z, h, a)
            sigma_next = torch.sqrt(var_next) * temperature
            
            z = sample_from_mdn(mu_next, sigma_next, pi_next)
            
            if step % 50 == 0:
                print(f"Dream step {step}/{num_steps}")
    
    return frames


def collect_real_frames(env, num_steps=500):
    """Collect frames from real environment with random actions."""
    frames = []
    obs, _ = env.reset()
    
    for step in range(num_steps):
        frames.append(obs.copy())
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
        
        if step % 50 == 0:
            print(f"Real step {step}/{num_steps}")
    
    return frames


def frames_to_mp4(frames, output_path="output.mp4", fps=30):
    """Convert frames to MP4."""
    import cv2
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Saved: {output_path} ({len(frames)} frames)")


def frames_to_gif(frames, output_path="output.gif", fps=30):
    """Convert frames to GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000/fps),
        loop=0
    )
    print(f"Saved: {output_path} ({len(frames)} frames)")


def create_side_by_side(real_frames, dream_frames, output_path="comparison.mp4", fps=30):
    """Create side-by-side comparison video."""
    import cv2
    
    n = min(len(real_frames), len(dream_frames))
    h, w = real_frames[0].shape[:2]
    
    # Resize dream frames to match real if needed
    dream_h, dream_w = dream_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w + dream_w + 10, max(h, dream_h)))
    
    for i in range(n):
        real = cv2.cvtColor(real_frames[i], cv2.COLOR_RGB2BGR)
        dream = cv2.cvtColor(dream_frames[i], cv2.COLOR_RGB2BGR)
        
        # Resize dream to match real height if needed
        if dream.shape[0] != real.shape[0]:
            dream = cv2.resize(dream, (int(dream_w * h / dream_h), h))
        
        gap = np.zeros((real.shape[0], 10, 3), dtype=np.uint8)
        combined = np.concatenate([real, gap, dream], axis=1)
        out.write(combined)
    
    out.release()
    print(f"Saved: {output_path}")


def main():
    # Initialize
    env = gym.make("CarRacing-v3")
    
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))
    print("Models loaded!")
    
    # Get initial observation
    initial_obs, _ = env.reset()
    
    # Generate dream
    print("\n=== Generating Dream Sequence ===")
    dream_frames = dream_rollout(
        vae, rnn, initial_obs, env.action_space,
        num_steps=300,
        temperature=1.0
    )
    
    # Collect real frames
    print("\n=== Collecting Real Frames ===")
    real_frames = collect_real_frames(env, num_steps=300)
    
    # Save outputs
    print("\n=== Saving Videos ===")
    frames_to_mp4(dream_frames, "dream.mp4", fps=30)
    frames_to_mp4(real_frames, "real.mp4", fps=30)
    create_side_by_side(real_frames, dream_frames, "comparison.mp4", fps=30)
    
    # Also save as GIF (smaller, easier to view)
    frames_to_gif(dream_frames[::3], "dream.gif", fps=10)  # Every 3rd frame for smaller file
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()