# evaluate_controller_vae.py

import torch
import numpy as np
import gymnasium as gym
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
import wandb
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    project="controller_vae_eval",
    config={
        "task": "Evaluate trained controller + VAE reconstructions",
        "environment": "CarRacing-v3",
    }
)


def run_controller_episode(vae, rnn, controller, env, max_steps=1000):
    """
    Run trained controller and collect frames + reconstructions.
    """
    vae.eval()
    rnn.eval()
    controller.eval()
    
    original_frames = []
    reconstructed_frames = []
    latent_vectors = []
    actions_taken = []
    rewards = []
    
    with torch.no_grad():
        obs, _ = env.reset()
        h = rnn.get_initial_hidden(device, batch_size=1)
        total_reward = 0
        
        for step in range(max_steps):
            # Save original frame
            original_frames.append(obs.copy())
            
            # Encode observation
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            # Save latent
            latent_vectors.append(z.squeeze(0).cpu().numpy())
            
            # Reconstruct frame using VAE
            reconstructed = vae.decode(z, 128)
            recon_img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_img = np.clip(recon_img, -1, 1)
            recon_img = ((recon_img + 1) / 2 * 255).astype(np.uint8)
            reconstructed_frames.append(recon_img)
            
            # Get action from controller
            h_for_controller = h[0][-1]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            action = controller(controller_input)
            action = torch.tanh(action)
            
            # Scale to CarRacing action space
            a = action.squeeze(0).cpu().numpy()
            a[1] = (a[1] + 1) / 2  # gas
            a[2] = (a[2] + 1) / 2  # brake
            a = a.astype(np.float32)
            
            actions_taken.append(a.copy())
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(a)
            total_reward += reward
            rewards.append(reward)
            
            # Update RNN hidden state
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            if step % 100 == 0:
                print(f"Step {step}/{max_steps}, Reward so far: {total_reward:.1f}")
            
            if done or truncated:
                print(f"Episode ended at step {step}")
                break
    
    return {
        'original': original_frames,
        'reconstructed': reconstructed_frames,
        'latents': np.array(latent_vectors),
        'actions': np.array(actions_taken),
        'rewards': np.array(rewards),
        'total_reward': total_reward
    }


def save_frames(frames, save_dir, prefix="frame"):
    """Save frames as images."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(save_dir, f"{prefix}_{idx:04d}.png"))
    print(f"Saved {len(frames)} frames to {save_dir}")


def log_to_wandb(results, num_samples=20):
    """Log results to wandb."""
    
    # Log total reward
    wandb.log({"total_reward": results['total_reward']})
    wandb.log({"episode_length": len(results['original'])})
    
    # Log reward over time
    cumulative_rewards = np.cumsum(results['rewards'])
    for step, (r, cr) in enumerate(zip(results['rewards'], cumulative_rewards)):
        wandb.log({
            "step": step,
            "reward": r,
            "cumulative_reward": cr
        })
    
    # Sample frames evenly throughout episode
    total_frames = len(results['original'])
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    # Create comparison gallery
    comparison_images = []
    for idx in indices:
        orig = results['original'][idx]
        recon = results['reconstructed'][idx]
        
        # Resize reconstructed to match original if needed
        if orig.shape[:2] != recon.shape[:2]:
            from PIL import Image
            recon_pil = Image.fromarray(recon)
            recon_pil = recon_pil.resize((orig.shape[1], orig.shape[0]))
            recon = np.array(recon_pil)
        
        # Create side-by-side comparison
        gap = np.ones((orig.shape[0], 10, 3), dtype=np.uint8) * 128
        combined = np.concatenate([orig, gap, recon], axis=1)
        
        comparison_images.append(
            wandb.Image(
                combined,
                caption=f"Step {idx} | Left: Original, Right: VAE Reconstruction"
            )
        )
    
    wandb.log({"reconstructions": comparison_images})
    
    # Log individual original and reconstructed galleries
    orig_gallery = [wandb.Image(results['original'][i], caption=f"Original step {i}") for i in indices]
    recon_gallery = [wandb.Image(results['reconstructed'][i], caption=f"Reconstructed step {i}") for i in indices]
    
    wandb.log({"original_frames": orig_gallery})
    wandb.log({"reconstructed_frames": recon_gallery})
    
    # Log actions distribution
    actions = results['actions']
    wandb.log({
        "steering_mean": actions[:, 0].mean(),
        "steering_std": actions[:, 0].std(),
        "gas_mean": actions[:, 1].mean(),
        "gas_std": actions[:, 1].std(),
        "brake_mean": actions[:, 2].mean(),
        "brake_std": actions[:, 2].std(),
    })
    
    # Create action histogram
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(actions[:, 0], bins=30, color='blue', alpha=0.7)
    axes[0].set_title('Steering Distribution')
    axes[0].set_xlabel('Steering')
    
    axes[1].hist(actions[:, 1], bins=30, color='green', alpha=0.7)
    axes[1].set_title('Gas Distribution')
    axes[1].set_xlabel('Gas')
    
    axes[2].hist(actions[:, 2], bins=30, color='red', alpha=0.7)
    axes[2].set_title('Brake Distribution')
    axes[2].set_xlabel('Brake')
    
    plt.tight_layout()
    wandb.log({"action_distributions": wandb.Image(fig)})
    plt.close()


def main():
    # Initialize environment
    env = gym.make("CarRacing-v3")
    
    # Load models
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    controller = Controller(input_features=32 + 35, actions_dims=3).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))
    controller.load_state_dict(torch.load("controller_final.pth", map_location=device))
    
    print("All models loaded!")
    
    # Run multiple episodes
    all_rewards = []
    
    for ep in range(5):
        print(f"\n=== Episode {ep + 1}/5 ===")
        
        results = run_controller_episode(vae, rnn, controller, env, max_steps=1000)
        all_rewards.append(results['total_reward'])
        
        # Save frames for first episode
        if ep == 0:
            save_frames(results['original'], "eval_frames/original")
            save_frames(results['reconstructed'], "eval_frames/reconstructed")
            
            # Log detailed results for first episode
            log_to_wandb(results, num_samples=30)
        
        print(f"Episode {ep + 1} reward: {results['total_reward']:.1f}")
        wandb.log({f"episode_{ep+1}_reward": results['total_reward']})
    
    # Log summary
    print(f"\n=== Summary ===")
    print(f"Rewards: {all_rewards}")
    print(f"Mean: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    
    wandb.log({
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "min_reward": np.min(all_rewards),
        "max_reward": np.max(all_rewards),
    })
    
    env.close()
    wandb.finish()
    print("\nDone! Check wandb for results.")


if __name__ == "__main__":
    main()