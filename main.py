from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v3")
training = Train(Env)

# Load your trained VAE weights
training.initialize(
    3, 3, 32, [64, 64, 128, 128], 3, 35, 5,
    hidden_layer=256,
    path_to_VAE_weights="vae_weights_epoch_05.pth"  # Use your best VAE checkpoint
)

# Collect latents (more rollouts here since RNN needs temporal data)
for i in range(5000):
    print(f"Latent rollout {i}")
    training.rollout(
        random_action=True,
        save_images=False,      # Don't need images anymore
        RNN_latents=True,       # Save latents, μ, logvar, actions
        save_root="rollouts_2", # Different directory
        max_steps=1000
    )

training.RNN_Train(epochs=20)