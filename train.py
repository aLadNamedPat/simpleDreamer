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
    project="new_project",

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
    ):
        self.vae = VAE(VAE_input_channels, VAE_out_channels, VAE_latent_dim, VAE_hidden_dims).to(device)
        self.rnn = RNN_MDN(VAE_latent_dim, action_dim, hidden_size, num_gaussians, hidden_layer, num_layers).to(device)
        self.controller = Controller(VAE_latent_dim, hidden_layer, self.env.action_space)

    def rollout(
        self,
        random_action = False,
        save_images : bool = False,
        save_root: str = "rollouts",
        max_steps: int = 1000,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_root, f"run_{timestamp}")
        if save_images:
            os.makedirs(save_dir, exist_ok=True)
            frames = []
            # actions = []
            # latents = []

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
                # z_np = z.squeeze(0).cpu().numpy()
                # latents.append(z_np.astype(np.float32))

                if random_action:
                    a = self.controller.random_action()
                else:
                    a = self.controller.step(torch.cat((z, h), dim  = 1))

                # a_np = a.astype(np.float32)
                # actions.append(a_np)

                obs, reward, done, _, _ = self.env.step(a)
                a = torch.from_numpy(a).unsqueeze(0).to(device)

                cumm_reward += reward
                (mu, var), h = self.rnn.forward(z, h, a)
                h = (h[0].detach(), h[1].detach())
                step += 1

        if save_images:
            # latents = np.stack(latents, axis=0)
            # actions = np.stack(actions, axis=0) 
            # np.savez_compressed(os.path.join(save_dir, "rollout_data.npz"), latents=latents, actions=actions)
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

        torch.save(self.vae.state_dict(), "vae_weights.pth")
                
    def RNN_Train(
        self,
        epochs,
    ):
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr= 0.0005)
        self.rnn.train()
        for epoch in range(epochs):
            train_loss = 0
            for imgs in self.loader():
                break
        return
