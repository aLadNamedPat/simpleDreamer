# visualize_trained_controller.py

import torch
import numpy as np
import os
from PIL import Image
import gymnasium as gym
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_from_mdn(mu, var, pi):
    """Sample from Mixture Density Network output."""
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


def get_action_from_controller(controller, z, h):
    """Get action from trained controller."""
    h_for_controller = h[0][-1]  # [1, hidden_size]
    controller_input = torch.cat([z, h_for_controller], dim=1)
    
    action = controller(controller_input)
    action = torch.tanh(action)
    
    # Scale to CarRacing action space
    a = action.squeeze(0).cpu().numpy()
    a[1] = (a[1] + 1) / 2  # gas: [-1,1] -> [0,1]
    a[2] = (a[2] + 1) / 2  # brake: [-1,1] -> [0,1]
    
    return a.astype(np.float32)


def rollout_real_env(vae, rnn, controller, env, num_steps=1000):
    """Run trained controller in real environment."""
    vae.eval()
    rnn.eval()
    controller.eval()
    
    frames = []
    total_reward = 0
    
    with torch.no_grad():
        obs, _ = env.reset()
        h = rnn.get_initial_hidden(device, batch_size=1)
        
        for step in range(num_steps):
            frames.append(obs.copy())
            
            # Encode observation
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            # Get action from trained controller
            a = get_action_from_controller(controller, z, h)
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(a)
            total_reward += reward
            
            # Update RNN hidden state
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            if step % 100 == 0:
                print(f"Real step {step}/{num_steps}, reward so far: {total_reward:.1f}")
            
            if done or truncated:
                print(f"Episode ended at step {step}")
                break
    
    print(f"Total reward: {total_reward:.1f}")
    return frames, total_reward


def rollout_dream(vae, rnn, controller, initial_obs, num_steps=500, temperature=1.0):
    """Run trained controller in dream environment."""
    vae.eval()
    rnn.eval()
    controller.eval()
    
    frames = []
    
    with torch.no_grad():
        # Encode initial observation
        obs_tensor = torch.from_numpy(initial_obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        mu, logvar = vae.encode(obs_tensor)
        z = vae.reparamterize(mu, logvar)
        
        h = rnn.get_initial_hidden(device, batch_size=1)
        
        for step in range(num_steps):
            # Decode current z to image for visualization
            reconstructed = vae.decode(z, 128)
            
            img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, -1, 1)
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            frames.append(img)
            
            # Get action from trained controller
            a = get_action_from_controller(controller, z, h)
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            
            # Predict next z using RNN
            (mu_next, var_next, pi_next), h = rnn.forward(z, h, a_tensor)
            
            # Sample next z
            z = sample_from_mdn(mu_next, var_next * (temperature ** 2), pi_next)
            
            if step % 100 == 0:
                print(f"Dream step {step}/{num_steps}")
    
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


def frames_to_gif(frames, output_path="output.gif", fps=15):
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
    real_h, real_w = real_frames[0].shape[:2]
    dream_h, dream_w = dream_frames[0].shape[:2]
    
    # Use max height, scale accordingly
    target_h = max(real_h, dream_h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Calculate widths after scaling
    real_w_scaled = int(real_w * target_h / real_h)
    dream_w_scaled = int(dream_w * target_h / dream_h)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (real_w_scaled + dream_w_scaled + 20, target_h))
    
    for i in range(n):
        real = cv2.cvtColor(real_frames[i], cv2.COLOR_RGB2BGR)
        dream = cv2.cvtColor(dream_frames[i], cv2.COLOR_RGB2BGR)
        
        # Resize to match height
        real = cv2.resize(real, (real_w_scaled, target_h))
        dream = cv2.resize(dream, (dream_w_scaled, target_h))
        
        # Add labels
        cv2.putText(real, "REAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(dream, "DREAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        gap = np.zeros((target_h, 20, 3), dtype=np.uint8)
        combined = np.concatenate([real, gap, dream], axis=1)
        out.write(combined)
    
    out.release()
    print(f"Saved: {output_path}")


def main():
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Load models
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    controller = Controller(input_features=32 + 35, actions_dims=3).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_04.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights_new/RNN_weights_best.pth", map_location=device))
    controller.load_state_dict(torch.load("controller_cma_gen_020.pth", map_location=device))
    
    print("All models loaded!")
    
    # Get initial observation for dream
    initial_obs, _ = env.reset()
    
    # === Real Environment Rollout ===
    print("\n=== Running Controller in Real Environment ===")
    real_frames, reward = rollout_real_env(vae, rnn, controller, env, num_steps=1000)
    
    # === Dream Environment Rollout ===
    print("\n=== Running Controller in Dream Environment ===")
    dream_frames = rollout_dream(vae, rnn, controller, initial_obs, num_steps=len(real_frames), temperature=1.0)
    
    # === Save Videos ===
    print("\n=== Saving Visualizations ===")
    
    # Individual videos
    frames_to_mp4(real_frames, "visualizations/real_trained.mp4", fps=30)
    frames_to_mp4(dream_frames, "visualizations/dream_trained.mp4", fps=30)
    
    # Side-by-side comparison
    create_side_by_side(real_frames, dream_frames, "visualizations/comparison_trained.mp4", fps=30)
    
    # GIFs (subsampled for smaller file size)
    frames_to_gif(real_frames[::2], "visualizations/real_trained.gif", fps=15)
    frames_to_gif(dream_frames[::2], "visualizations/dream_trained.gif", fps=15)
    
    # === Run Multiple Episodes ===
    print("\n=== Running 5 More Episodes ===")
    all_rewards = [reward]
    
    for ep in range(5):
        frames, r = rollout_real_env(vae, rnn, controller, env, num_steps=1000)
        all_rewards.append(r)
        frames_to_mp4(frames, f"visualizations/episode_{ep+1}.mp4", fps=30)
    
    print(f"\n=== Summary ===")
    print(f"Rewards: {all_rewards}")
    print(f"Mean: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    
    env.close()
    print("\nDone! Check the 'visualizations/' folder.")


if __name__ == "__main__":
    main()