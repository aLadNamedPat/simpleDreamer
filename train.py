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

wandb.init(
    # set the wandb project where this run will be logged
    project="RNN_train",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "CNN",
    "dataset": "Car-Racer-V2",
    "batch_size" : 64,
    "latent_dims" : 32,
    "epochs": 10,
    }
)

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
        self.controller = Controller(VAE_latent_dim, hidden_layer, self.env.action_space)
        if path_to_VAE_weights is not None:
            state_dict = torch.load(path_to_VAE_weights, map_location=device)
            self.vae.load_state_dict(state_dict)
            self.vae.to(device)

        if path_to_RNN_weights is not None:
            state_dict = torch.load(path_to_RNN_weights, map_location=device)
            self.rnn.load_state_dict(state_dict)
            self.rnn.to(device)

    def rollout(
        self,
        random_action = False,
        save_images : bool = False,
        RNN_latents : bool = False,
        save_root: str = "rollouts",
        max_steps: int = 1000,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_root, f"run_{timestamp}")
        if save_images:
            os.makedirs(save_dir, exist_ok=True)
            frames = []
        
        if RNN_latents:
            if not save_images:
                os.makedirs(save_dir, exist_ok=True)
            actions = []
            latents = []

        obs = self.env.reset()[0]
        h = self.rnn.get_initial_hidden(device)
        print(h)
        cumm_reward = 0
        step = 0
        done = False

        with torch.no_grad():
            while not done and step < max_steps:
                if save_images:
                    frames.append(obs.copy())
                obs_tensor = torch.from_numpy(obs)
                obs_tensor = obs_tensor.to(torch.float32) / 255
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                u, var = self.vae.encode(obs_tensor)

                z = self.vae.reparamterize(u, var).to(device)
                if random_action:
                    a = self.controller.random_action()
                else:
                    a = self.controller.step(torch.cat((z, h), dim  = 1))

                if RNN_latents:
                    z_np = z.squeeze(0).cpu().numpy()
                    latents.append(z_np.astype(np.float32))
                    a_np = a.astype(np.float32)
                    actions.append(a_np)

                obs, reward, done, _, _ = self.env.step(a)
                a = torch.from_numpy(a).unsqueeze(0).to(device)

                cumm_reward += reward
                (mu, var), h = self.rnn.forward(z, h, a)
                h = (h[0].detach(), h[1].detach())
                step += 1

        if RNN_latents:
            latents = np.stack(latents, axis=0)
            actions = np.stack(actions, axis=0) 
            np.savez_compressed(os.path.join(save_dir, "rollout_data.npz"), latents=latents, actions=actions)

        if save_images:
            for idx, frame in enumerate(frames):
                # if your obs is float [0,1], convert back to uint8:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                img = Image.fromarray(frame)
                img.save(os.path.join(save_dir, f"frame_{idx:04d}.png"))

        return cumm_reward
    
    def VAE_Train(
        self,
        epochs,
        batch_size : int = 150,
        kld_weight = 0.00025,
    ):
        dataset = RolloutImageDataset(root_dir="rollouts", img_size=96)
        self.loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr= 0.0005)
        self.vae.train()
        for epoch in range(epochs):
            train_loss = 0
            for imgs in tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                imgs = imgs.to(device)
                recon, mu, var = self.vae(imgs)
                loss = self.vae.find_loss(recon, imgs, mu, var, kld_weight)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                wandb.log({"loss": loss})
            vis_in   = imgs[:8].cpu()
            vis_out  = recon[:8].cpu()
            gallery = []
            for i in range(vis_in.size(0)):
                gallery.append(
                    wandb.Image(
                        torch.cat([vis_in[i], vis_out[i]], dim=2),
                        caption=f"E{epoch}_idx{i}"
                    )
                )
            wandb.log({"recon_gallery": gallery, "epoch": epoch})
            mean_loss = train_loss / len(self.loader)
            wandb.log({"epoch_loss": mean_loss, "epoch": epoch})
            save_path = f"vae_weights_epoch_{epoch+1:02d}.pth"
            torch.save(self.vae.state_dict(), save_path)
            print(f"→ Saved VAE weights to {save_path}")
                
    def RNN_Train(
        self,
        epochs,
        batch_size  = 16
    ):
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr= 0.0001)
        self.rnn.train()
        self.loader = RolloutLatentDataset(root_dir="rollouts_2", segment_len=128)

        dataloader = DataLoader(self.loader,
                        batch_size=batch_size,
                        drop_last=True) 
        os.makedirs("weights", exist_ok=True)

        for epoch in range(epochs):
            h = None
            total_loss = 0
            for x, a, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, a, y = x.to(device), a.to(device), y.to(device)
                
                loss, h = self.rnn.MDN_loss(torch.cat((x, a), dim = -1), y.unsqueeze(2), h)  # y→[B,T,1,D]
                h = (h[0].detach(), h[1].detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                print(total_loss)
                wandb.log({"loss": loss})

            save_path = f"weights/RNN_weights_epoch_{epoch+1:02d}.pth"
            torch.save(self.rnn.state_dict(), save_path)
