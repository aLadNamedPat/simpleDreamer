from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
from Loader import *
import torch
import os
import time
from PIL import Image
import numpy as np
import wandb
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from BrownianActionSampler import BrownianActionSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train():
    def __init__(
        self,
        Env,
    ):
        self.env = Env

    def initialize(
        self,
        VAE_input_channels : int, 
        VAE_out_channels : int, 
        VAE_latent_dim : int, 
        VAE_hidden_dims : list,
        action_dim : int,
        hidden_size : int,
        num_gaussians : int,
        hidden_layer : int = 40,
        num_layers : int = 1,
        path_to_VAE_weights : str = None,
        path_to_RNN_weights : str = None,
    ):
        self.vae = VAE(VAE_input_channels, VAE_out_channels, VAE_latent_dim, VAE_hidden_dims).to(device)
        self.rnn = RNN_MDN(VAE_latent_dim, action_dim, hidden_size, num_gaussians, hidden_layer, num_layers).to(device)
        self.controller = Controller(VAE_latent_dim, hidden_size, self.env.action_space)
        if path_to_VAE_weights is not None:
            state_dict = torch.load(path_to_VAE_weights, map_location=device)
            self.vae.load_state_dict(state_dict)
            self.vae.to(device)

        if path_to_RNN_weights is not None:
            state_dict = torch.load(path_to_RNN_weights, map_location=device)
            self.rnn.load_state_dict(state_dict)
            self.rnn.to(device)
    
    def collect_rollouts(
        self,
        num_rollouts: int = 5000,
        save_root: str = "unified_rollouts",
        max_steps: int = 1000,
        dt: float = 0.1,
        volatility: np.ndarray = None,
        use_forward_bias: bool = True,
        gas_bias: float = 0.2,
    ):
        os.makedirs(save_root, exist_ok=True)
        self.action_sampler = BrownianActionSampler(
            self.env.action_space,
            dt=dt,
            volatility=volatility,
        )
        
        print("=" * 60)
        print(f"Collecting {num_rollouts} rollouts with Brownian motion actions")
        print("=" * 60)
        print(f"Save directory: {save_root}")
        print(f"Max steps per rollout: {max_steps}")
        print(f"Brownian dt: {dt}")
        print(f"Volatility: {self.action_sampler.volatility}")
        print(f"Forward bias: {use_forward_bias} (gas_bias={gas_bias})")
        print("=" * 60)
        
        results = []
        total_steps = 0
        total_reward = 0
        
        for i in tqdm(range(num_rollouts), desc="Collecting rollouts"):
            result = self._collect_single_rollout(
                save_root=save_root,
                max_steps=max_steps,
                use_forward_bias=use_forward_bias,
                gas_bias=gas_bias,
            )
            results.append(result)
            total_steps += result["num_steps"]
            total_reward += result["cumulative_reward"]
        
        # Summary
        print("\n" + "=" * 60)
        print("Collection Complete!")
        print("=" * 60)
        print(f"Total rollouts: {num_rollouts}")
        print(f"Total frames: {total_steps}")
        print(f"Average steps/rollout: {total_steps / num_rollouts:.1f}")
        print(f"Average reward: {total_reward / num_rollouts:.1f}")
        print(f"Data saved to: {save_root}")
        print("=" * 60)
        
        return results
    
    def _collect_single_rollout(
        self,
        save_root: str,
        max_steps: int,
        use_forward_bias: bool,
        gas_bias: float,
    ) -> dict:
        """
        Collect a single rollout.
        
        Saves:
        - images/frame_XXXX.png: Raw frames for VAE training
        - rollout_data.npz: Actions, rewards, metadata (latents added later)
        """
        # Create save directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{np.random.randint(10000):04d}"
        save_dir = os.path.join(save_root, run_id)
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Reset environment and sampler
        obs, _ = self.env.reset()
        self.action_sampler.reset()
        
        # Storage
        frames = []
        actions = []
        rewards = []
        
        cumm_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Store frame
            frames.append(obs.copy())
            
            # Get action from Brownian sampler
            if use_forward_bias:
                a = self.action_sampler.sample_with_forward_bias(gas_bias)
            else:
                a = self.action_sampler.sample()
            
            actions.append(a.copy())
            
            # Step environment
            obs, reward, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            
            rewards.append(reward)
            cumm_reward += reward
            step += 1
        
        # Convert to numpy arrays
        actions = np.stack(actions, axis=0).astype(np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        
        # Save images
        for idx, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(images_dir, f"frame_{idx:04d}.png"))
        
        # Save rollout data (latents will be added later by encode_rollouts)
        np.savez_compressed(
            os.path.join(save_dir, "rollout_data.npz"),
            actions=actions,
            rewards=rewards,
            num_steps=np.array(step),
            cumulative_reward=np.array(cumm_reward),
            timestamp=timestamp,
        )
        
        return {
            "run_id": run_id,
            "save_dir": save_dir,
            "num_steps": step,
            "cumulative_reward": cumm_reward,
        }

    def VAE_Train(
        self,
        epochs: int,
        batch_size: int = 150,
        kld_weight: float = 0.00025,
        learning_rate: float = 0.0005,
        data_root: str = "unified_rollouts",
        project_name: str = "VAE_train",
    ):
        """
        Train VAE on collected images.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            kld_weight: KL divergence weight
            learning_rate: Learning rate
            data_root: Root directory containing rollouts
            project_name: wandb project name
        """
        wandb.init(
            project=project_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "kld_weight": kld_weight,
                "epochs": epochs,
                "data_root": data_root,
            }
        )
        
        # Create dataset from unified rollouts
        dataset = RolloutImageDataset(root_dir=data_root, img_size=96)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        print(f"Training VAE on {len(dataset)} images from {data_root}")
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.vae.train()
        
        for epoch in range(epochs):
            train_loss = 0
            for imgs in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                imgs = imgs.to(device)
                recon, mu, var = self.vae(imgs)
                loss = self.vae.find_loss(recon, imgs, mu, var, kld_weight)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                wandb.log({"loss": loss})
            
            # Log reconstructions
            vis_in = imgs[:8].cpu()
            vis_out = recon[:8].cpu()
            gallery = []
            for i in range(vis_in.size(0)):
                gallery.append(
                    wandb.Image(
                        torch.cat([vis_in[i], vis_out[i]], dim=2),
                        caption=f"E{epoch}_idx{i}"
                    )
                )
            
            mean_loss = train_loss / len(loader)
            wandb.log({
                "recon_gallery": gallery,
                "epoch_loss": mean_loss,
                "epoch": epoch,
            })
            
            # Save weights
            save_path = f"vae_weights_epoch_{epoch+1:02d}.pth"
            torch.save(self.vae.state_dict(), save_path)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {mean_loss:.4f} | Saved to {save_path}")
        
        wandb.finish()

    def encode_rollouts(
        self,
        data_root: str = "unified_rollouts",
        batch_size: int = 64,
    ):
        """
        Encode all rollout images to latents using trained VAE.
        
        Call this AFTER VAE training to prepare data for RNN-MDN training.
        
        Adds to each rollout's npz file:
        - 'latents': sampled z values [T, latent_dim]
        - 'mu': encoder means [T, latent_dim]
        - 'logvar': encoder log-variances [T, latent_dim]
        
        Args:
            data_root: Root directory containing rollouts
            batch_size: Batch size for encoding
        """
        print("=" * 60)
        print("Encoding rollout images to latents")
        print("=" * 60)
        
        self.vae.eval()
        
        # Find all rollout directories
        rollout_dirs = sorted(glob.glob(os.path.join(data_root, "run_*")))
        print(f"Found {len(rollout_dirs)} rollouts to encode")
        
        encoded_count = 0
        skipped_count = 0
        
        for rollout_dir in tqdm(rollout_dirs, desc="Encoding rollouts"):
            success = self._encode_single_rollout(rollout_dir, batch_size)
            if success:
                encoded_count += 1
            else:
                skipped_count += 1
        
        print("\n" + "=" * 60)
        print("Encoding Complete!")
        print("=" * 60)
        print(f"Encoded: {encoded_count}")
        print(f"Skipped: {skipped_count}")
        print("=" * 60)
    
    def _encode_single_rollout(
        self,
        rollout_dir: str,
        batch_size: int,
    ) -> bool:
        """
        Encode a single rollout's images to latents.
        
        Returns True if encoding was performed, False if skipped.
        """
        images_dir = os.path.join(rollout_dir, "images")
        npz_path = os.path.join(rollout_dir, "rollout_data.npz")
        
        if not os.path.exists(images_dir):
            return False
        
        # Load existing data
        existing_data = dict(np.load(npz_path, allow_pickle=True))
        
        # Check if already encoded
        if 'latents' in existing_data:
            return False
        
        # Load and sort image files
        image_files = sorted(glob.glob(os.path.join(images_dir, "frame_*.png")))
        
        if len(image_files) == 0:
            return False
        
        latents = []
        mus = []
        logvars = []
        
        with torch.no_grad():
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                
                # Load batch
                batch_images = []
                for img_path in batch_files:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                    batch_images.append(img_tensor)
                
                batch_tensor = torch.stack(batch_images).to(device)
                
                # Encode
                mu, logvar = self.vae.encode(batch_tensor)
                z = self.vae.reparamterize(mu, logvar)
                
                latents.append(z.cpu().numpy())
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
        
        # Concatenate batches
        latents = np.concatenate(latents, axis=0).astype(np.float32)
        mus = np.concatenate(mus, axis=0).astype(np.float32)
        logvars = np.concatenate(logvars, axis=0).astype(np.float32)
        
        # Update and save
        existing_data['latents'] = latents
        existing_data['mu'] = mus
        existing_data['logvar'] = logvars
        
        np.savez_compressed(npz_path, **existing_data)
        
        return True
    
    def RNN_Train(
        self,
        epochs: int,
        batch_size: int = 16,
        seq_len: int = 32,
        stride: int = 2,
        test_count: int = 100,
        eval_every: int = 1,
        data_root: str = "unified_rollouts",
        project_name: str = "RNN_train",
    ):
        """
        Train RNN-MDN on encoded latent sequences.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            seq_len: Sequence length for training
            stride: Stride for sequence sampling
            test_count: Number of rollouts to reserve for test set
            eval_every: Evaluate on test set every N epochs
            data_root: Root directory containing encoded rollouts
            project_name: wandb project name
        """
        wandb.init(
            project=project_name,
            config={
                "learning_rate": 0.001,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "stride": stride,
                "epochs": epochs,
                "data_root": data_root,
            }
        )
        
        initial_lr = 0.001
        min_lr = 0.0001
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=initial_lr)
        
        # Create train and test dataloaders
        train_loader, test_loader = get_train_test_loaders(
            root_dir=data_root,
            seq_len=seq_len,
            sample_latent=True,
            stride=stride,
            test_count=test_count,
            batch_size=batch_size,
            num_workers=4,
        )
        
        print(f"Training RNN-MDN on data from {data_root}")
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        os.makedirs("weights_rnn", exist_ok=True)
        
        best_test_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.rnn.train()
            total_train_loss = 0
            num_train_batches = 0
            
            for x, a, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                x, a, y = x.to(device), a.to(device), y.to(device)
                
                rnn_input = torch.cat((x, a), dim=-1)
                loss, _ = self.rnn.MDN_loss(rnn_input, y, h0=None)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = total_train_loss / num_train_batches
            
            # Evaluation
            if (epoch + 1) % eval_every == 0:
                self.rnn.eval()
                total_test_loss = 0
                num_test_batches = 0
                
                with torch.no_grad():
                    for x, a, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False):
                        x, a, y = x.to(device), a.to(device), y.to(device)
                        
                        rnn_input = torch.cat((x, a), dim=-1)
                        loss, _ = self.rnn.MDN_loss(rnn_input, y, h0=None)
                        
                        total_test_loss += loss.item()
                        num_test_batches += 1
                
                avg_test_loss = total_test_loss / num_test_batches
                
                wandb.log({
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                })
                
                print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | "
                      f"Test: {avg_test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    torch.save(self.rnn.state_dict(), "weights_rnn/RNN_weights_best.pth")
                    print(f"  → New best test loss! Saved best model.")
            else:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                })
                print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            scheduler.step()
            
            save_path = f"weights_rnn/RNN_weights_epoch_{epoch+1:02d}.pth"
            torch.save(self.rnn.state_dict(), save_path)
        
        wandb.finish()
        print(f"\nRNN training complete! Best test loss: {best_test_loss:.4f}")

    def rollout(
        self,
        random_action: bool = False,
        use_brownian: bool = True,
        save_images: bool = False,
        RNN_latents: bool = False,
        save_root: str = "rollouts",
        max_steps: int = 1000,
    ):
        """
        Run a single rollout (for evaluation or visualization).
        
        Args:
            random_action: Use random actions (legacy, uses uniform random)
            use_brownian: Use Brownian motion for random actions (recommended)
            save_images: Save frame images
            RNN_latents: Save latent data for RNN (requires trained VAE)
            save_root: Directory to save data
            max_steps: Maximum steps
        
        Returns:
            Cumulative reward
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_root, f"run_{timestamp}")
        
        if save_images or RNN_latents:
            os.makedirs(save_dir, exist_ok=True)
        
        if save_images:
            frames = []
        
        if RNN_latents:
            actions = []
            latents = []
            mus = []
            variances = []
        
        obs, _ = self.env.reset()
        h = self.rnn.get_initial_hidden(device)
        
        if use_brownian and self.action_sampler is not None:
            self.action_sampler.reset()
        
        cumm_reward = 0
        step = 0
        done = False
        
        with torch.no_grad():
            while not done and step < max_steps:
                if save_images:
                    frames.append(obs.copy())
                
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                u, var = self.vae.encode(obs_tensor)
                z = self.vae.reparamterize(u, var).to(device)
                
                # Get action
                if random_action:
                    if use_brownian and self.action_sampler is not None:
                        a = self.action_sampler.sample_with_forward_bias()
                    else:
                        a = self.controller.random_action()
                else:
                    # Use controller with z and RNN hidden state
                    h_flat = h[0][-1]  # Get last layer hidden state
                    a = self.controller.step(torch.cat((z, h_flat), dim=1))
                
                if RNN_latents:
                    latents.append(z.squeeze(0).cpu().numpy().astype(np.float32))
                    mus.append(u.squeeze(0).cpu().numpy().astype(np.float32))
                    variances.append(var.squeeze(0).cpu().numpy().astype(np.float32))
                    actions.append(a.astype(np.float32))
                
                obs, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                
                a_tensor = torch.from_numpy(a).unsqueeze(0).to(device)
                cumm_reward += reward
                
                (mu_rnn, var_rnn, pi), h = self.rnn.forward(z, h, a_tensor)
                h = (h[0].detach(), h[1].detach())
                step += 1
        
        if RNN_latents:
            np.savez_compressed(
                os.path.join(save_dir, "rollout_data.npz"),
                latents=np.stack(latents, axis=0),
                actions=np.stack(actions, axis=0),
                mu=np.stack(mus, axis=0),
                logvar=np.stack(variances, axis=0),
            )
        
        if save_images:
            for idx, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(save_dir, f"frame_{idx:04d}.png"))
        
        return cumm_reward