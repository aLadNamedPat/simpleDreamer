# visualize_rnn_predictions.py
"""
Sample one episode from saved rollouts, generate dream predictions alongside
actual states, and log to wandb for visualization.
"""

import torch
import numpy as np
import os
import glob
import random
import wandb
from PIL import Image

from VAE import VAE
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_from_mdn(mu, var, pi, temperature=1.0):
    """
    Sample from Mixture Density Network output.
    
    Args:
        mu:  [1, n_gaussians, latent_dim]
        var: [1, n_gaussians, latent_dim]
        pi:  [1, n_gaussians, latent_dim]
        temperature: sampling temperature (higher = more random)
    
    Returns:
        z_next: [1, latent_dim]
    """
    mu = mu.squeeze(0)      # [n_gaussians, latent_dim]
    sigma = torch.sqrt(var).squeeze(0) * temperature
    pi = pi.squeeze(0)      # [n_gaussians, latent_dim]
    
    n_gaussians, latent_dim = mu.shape
    
    # Numerical stability
    pi = torch.clamp(pi, min=1e-8)
    pi = pi / pi.sum(dim=0, keepdim=True)
    
    if torch.isnan(pi).any() or torch.isinf(pi).any():
        pi = torch.ones_like(pi) / n_gaussians
    
    # Sample component for each latent dimension
    pi_t = pi.permute(1, 0)  # [latent_dim, n_gaussians]
    indices = torch.multinomial(pi_t, num_samples=1).squeeze(-1)  # [latent_dim]
    
    # Gather selected parameters
    latent_indices = torch.arange(latent_dim, device=mu.device)
    mu_sel = mu[indices, latent_indices]
    sigma_sel = sigma[indices, latent_indices]
    
    # Sample
    z_next = mu_sel + sigma_sel * torch.randn_like(mu_sel)
    
    return z_next.unsqueeze(0)


def get_mdn_mean(mu, pi):
    """Get weighted mean prediction from MDN (deterministic)."""
    mu = mu.squeeze(0)  # [n_gaussians, latent_dim]
    pi = pi.squeeze(0)  # [n_gaussians, latent_dim]
    
    # Weighted sum across gaussians
    z_mean = (pi * mu).sum(dim=0)  # [latent_dim]
    return z_mean.unsqueeze(0)


def load_random_episode(rollouts_dir="rollouts_rnn"):
    """Load a random episode from saved rollouts."""
    npz_files = sorted(glob.glob(os.path.join(rollouts_dir, "run_*", "rollout_data.npz")))
    
    if not npz_files:
        raise RuntimeError(f"No rollout files found in {rollouts_dir}")
    
    # Pick a random episode
    chosen_file = random.choice(npz_files)
    print(f"Selected episode: {chosen_file}")
    
    data = np.load(chosen_file)
    
    episode = {
        'mu': data['mu'],           # [T, z_dim]
        'logvar': data['logvar'],   # [T, z_dim]
        'actions': data['actions'], # [T, a_dim]
    }
    
    # Also get the frame directory for actual images
    run_dir = os.path.dirname(chosen_file)
    frame_files = sorted(glob.glob(os.path.join(run_dir, "frame_*.png")))
    episode['frame_files'] = frame_files
    episode['run_dir'] = run_dir
    
    print(f"Episode length: {len(episode['actions'])} steps")
    print(f"Found {len(frame_files)} frame images")
    
    return episode


def decode_latent_to_image(vae, z, img_size=128):
    """Decode latent vector to RGB image."""
    with torch.no_grad():
        reconstructed = vae.decode(z, img_size)
        img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, -1, 1)
        img = ((img + 1) / 2 * 255).astype(np.uint8)
    return img


def create_comparison_image(actual_img, predicted_img, onestep_img=None):
    """Create side-by-side comparison image."""
    h, w = actual_img.shape[:2]
    
    if onestep_img is not None:
        # Three images: actual | one-step | dream
        gap = np.ones((h, 10, 3), dtype=np.uint8) * 128
        combined = np.concatenate([actual_img, gap, onestep_img, gap, predicted_img], axis=1)
    else:
        # Two images: actual | dream
        gap = np.ones((h, 10, 3), dtype=np.uint8) * 128
        combined = np.concatenate([actual_img, gap, predicted_img], axis=1)
    
    return combined


def run_visualization(
    vae_path="vae_weights_epoch_04.pth",
    rnn_path="weights_new/RNN_weights_best.pth",
    rollouts_dir="rollouts_rnn",
    num_steps=200,
    temperature=1.0,
    log_every=5,
    use_stochastic=True,
    project_name="world-model-viz",
    run_name=None,
):
    """
    Main visualization function.
    
    Args:
        vae_path: Path to VAE weights
        rnn_path: Path to RNN weights
        rollouts_dir: Directory containing rollout data
        num_steps: Number of steps to visualize
        temperature: MDN sampling temperature
        log_every: Log to wandb every N steps
        use_stochastic: Use stochastic sampling (True) or mean prediction (False)
        project_name: wandb project name
        run_name: wandb run name (auto-generated if None)
    """
    
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name or f"rnn-viz-temp{temperature}",
        config={
            "temperature": temperature,
            "num_steps": num_steps,
            "use_stochastic": use_stochastic,
            "vae_path": vae_path,
            "rnn_path": rnn_path,
        }
    )
    
    # Load models
    print("Loading models...")
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    
    vae.eval()
    rnn.eval()
    
    print("Models loaded!")
    
    # Load episode
    episode = load_random_episode(rollouts_dir)
    
    mu_all = torch.from_numpy(episode['mu']).float().to(device)
    logvar_all = torch.from_numpy(episode['logvar']).float().to(device)
    actions_all = torch.from_numpy(episode['actions']).float().to(device)
    frame_files = episode['frame_files']
    
    T = min(num_steps, len(actions_all) - 1)
    
    # Initialize
    with torch.no_grad():
        # Get initial latent from actual data
        z_actual = mu_all[0:1]  # [1, z_dim] - use mean as initial state
        z_dream = z_actual.clone()
        
        # Initialize RNN hidden states
        h_onestep = rnn.get_initial_hidden(device, batch_size=1)
        h_dream = rnn.get_initial_hidden(device, batch_size=1)
        
        # Storage for metrics
        latent_errors = []
        
        # Collect frames for video
        comparison_frames = []
        
        print(f"\nRunning {T} step visualization...")
        
        for t in range(T):
            # === Get actual latent and image ===
            z_actual = mu_all[t:t+1]  # [1, z_dim]
            action = actions_all[t:t+1]  # [1, a_dim]
            
            # Decode actual latent to image
            actual_img = decode_latent_to_image(vae, z_actual)
            
            # Load actual frame if available
            if t < len(frame_files):
                actual_frame = np.array(Image.open(frame_files[t]).convert('RGB'))
            else:
                actual_frame = actual_img
            
            # === One-step prediction (reset hidden each step) ===
            (mu_pred, var_pred, pi_pred), h_onestep = rnn.forward(z_actual, h_onestep, action)
            
            if use_stochastic:
                z_onestep_pred = sample_from_mdn(mu_pred, var_pred, pi_pred, temperature)
            else:
                z_onestep_pred = get_mdn_mean(mu_pred, pi_pred)
            
            onestep_img = decode_latent_to_image(vae, z_onestep_pred)
            
            # === Dream rollout (continuous, no reset) ===
            dream_img = decode_latent_to_image(vae, z_dream)
            
            # Predict next dream state
            (mu_dream, var_dream, pi_dream), h_dream = rnn.forward(z_dream, h_dream, action)
            
            if use_stochastic:
                z_dream = sample_from_mdn(mu_dream, var_dream, pi_dream, temperature)
            else:
                z_dream = get_mdn_mean(mu_dream, pi_dream)
            
            # === Compute metrics ===
            # Latent space error (dream vs actual)
            z_actual_next = mu_all[t+1:t+2] if t+1 < len(mu_all) else z_actual
            latent_error = torch.mean((z_dream - z_actual_next) ** 2).item()
            latent_errors.append(latent_error)
            
            # One-step prediction error
            onestep_error = torch.mean((z_onestep_pred - z_actual_next) ** 2).item()
            
            # === Create comparison image ===
            comparison = create_comparison_image(actual_frame, dream_img, onestep_img)
            comparison_frames.append(comparison)
            
            # === Log to wandb ===
            if t % log_every == 0:
                wandb.log({
                    "step": t,
                    "latent_error_dream": latent_error,
                    "latent_error_onestep": onestep_error,
                    "cumulative_dream_error": np.mean(latent_errors),
                    "comparison": wandb.Image(
                        comparison, 
                        caption=f"Step {t}: Actual | One-step | Dream"
                    ),
                    "actual": wandb.Image(actual_frame, caption=f"Actual frame {t}"),
                    "dream": wandb.Image(dream_img, caption=f"Dream frame {t}"),
                    "onestep": wandb.Image(onestep_img, caption=f"One-step pred {t}"),
                })
            
            if t % 50 == 0:
                print(f"Step {t}/{T} | Dream error: {latent_error:.4f} | "
                      f"One-step error: {onestep_error:.4f}")
        
        # === Log video ===
        print("\nCreating video...")
        video_array = np.stack(comparison_frames)  # [T, H, W, C]
        video_array = video_array.transpose(0, 3, 1, 2)  # [T, C, H, W]
        
        wandb.log({
            "comparison_video": wandb.Video(video_array, fps=15, format="mp4"),
            "final_cumulative_error": np.mean(latent_errors),
            "final_step_error": latent_errors[-1] if latent_errors else 0,
        })
        
        # === Summary metrics ===
        print(f"\n=== Summary ===")
        print(f"Total steps: {T}")
        print(f"Mean latent error: {np.mean(latent_errors):.4f}")
        print(f"Final latent error: {latent_errors[-1]:.4f}")
        print(f"Max latent error: {np.max(latent_errors):.4f}")
        
        # Log summary
        wandb.summary["mean_latent_error"] = np.mean(latent_errors)
        wandb.summary["final_latent_error"] = latent_errors[-1]
        wandb.summary["max_latent_error"] = np.max(latent_errors)
    
    wandb.finish()
    print("\nDone! Check wandb for visualizations.")


def run_multi_episode_comparison(
    vae_path="vae_weights_epoch_04.pth",
    rnn_path="weights_new/RNN_weights_best.pth",
    rollouts_dir="rollouts_rnn",
    num_episodes=5,
    num_steps=200,
    temperature=1.0,
    project_name="world-model-viz",
):
    """Run visualization on multiple episodes and compare."""
    
    wandb.init(
        project=project_name,
        name=f"multi-episode-comparison",
        config={
            "num_episodes": num_episodes,
            "num_steps": num_steps,
            "temperature": temperature,
        }
    )
    
    # Load models
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    
    vae.eval()
    rnn.eval()
    
    # Get all episodes
    npz_files = sorted(glob.glob(os.path.join(rollouts_dir, "run_*", "rollout_data.npz")))
    selected_files = random.sample(npz_files, min(num_episodes, len(npz_files)))
    
    all_errors = []
    
    for ep_idx, npz_file in enumerate(selected_files):
        print(f"\n=== Episode {ep_idx + 1}/{len(selected_files)} ===")
        print(f"File: {npz_file}")
        
        data = np.load(npz_file)
        mu_all = torch.from_numpy(data['mu']).float().to(device)
        actions_all = torch.from_numpy(data['actions']).float().to(device)
        
        T = min(num_steps, len(actions_all) - 1)
        
        with torch.no_grad():
            z_dream = mu_all[0:1].clone()
            h_dream = rnn.get_initial_hidden(device, batch_size=1)
            
            episode_errors = []
            
            for t in range(T):
                action = actions_all[t:t+1]
                
                # Dream prediction
                (mu_dream, var_dream, pi_dream), h_dream = rnn.forward(z_dream, h_dream, action)
                z_dream = sample_from_mdn(mu_dream, var_dream, pi_dream, temperature)
                
                # Error vs actual
                z_actual_next = mu_all[t+1:t+2] if t+1 < len(mu_all) else mu_all[t:t+1]
                error = torch.mean((z_dream - z_actual_next) ** 2).item()
                episode_errors.append(error)
            
            all_errors.append(episode_errors)
            
            wandb.log({
                f"episode_{ep_idx}_mean_error": np.mean(episode_errors),
                f"episode_{ep_idx}_final_error": episode_errors[-1],
            })
    
    # Aggregate statistics
    all_errors = np.array(all_errors)  # [num_episodes, T]
    mean_error_per_step = np.mean(all_errors, axis=0)
    std_error_per_step = np.std(all_errors, axis=0)
    
    # Log error over time
    for t in range(len(mean_error_per_step)):
        wandb.log({
            "step": t,
            "mean_error": mean_error_per_step[t],
            "std_error": std_error_per_step[t],
        })
    
    wandb.summary["overall_mean_error"] = np.mean(all_errors)
    wandb.summary["overall_std_error"] = np.std(all_errors)
    
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize RNN predictions vs actual states")
    parser.add_argument("--vae", default="vae_weights_epoch_04.pth", help="VAE weights path")
    parser.add_argument("--rnn", default="weights_new/RNN_weights_best.pth", help="RNN weights path")
    parser.add_argument("--rollouts", default="rollouts_rnn", help="Rollouts directory")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to visualize")
    parser.add_argument("--temperature", type=float, default=1.0, help="MDN sampling temperature")
    parser.add_argument("--log-every", type=int, default=5, help="Log to wandb every N steps")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (mean) predictions")
    parser.add_argument("--project", default="world-model-viz", help="wandb project name")
    parser.add_argument("--multi", action="store_true", help="Run multi-episode comparison")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes for multi comparison")
    
    args = parser.parse_args()
    
    if args.multi:
        run_multi_episode_comparison(
            vae_path=args.vae,
            rnn_path=args.rnn,
            rollouts_dir=args.rollouts,
            num_episodes=args.num_episodes,
            num_steps=args.steps,
            temperature=args.temperature,
            project_name=args.project,
        )
    else:
        run_visualization(
            vae_path=args.vae,
            rnn_path=args.rnn,
            rollouts_dir=args.rollouts,
            num_steps=args.steps,
            temperature=args.temperature,
            log_every=args.log_every,
            use_stochastic=not args.deterministic,
            project_name=args.project,
        )