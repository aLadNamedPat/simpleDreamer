from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v3")
training = Train(Env)

# Initialize with trained VAE weights (needed to encode images to latents)
training.initialize(
    3, 3, 32, [64, 64, 128, 128], 3, 35, 5,
    hidden_layer=256,
    path_to_VAE_weights="vae_weights_epoch_04.pth",  # Load your trained VAE
    path_to_RNN_weights=None
)

# # Collect latent rollouts for RNN training
# for i in range(5000):  # Fewer rollouts needed since each has many timesteps
#     print(f"Latent rollout {i}")
#     training.rollout(
#         random_action=True,
#         save_images=False,      # Don't need raw images
#         RNN_latents=True,       # Save latent encodings + actions
#         save_root="rollouts_rnn",  # Different directory for RNN data
#         max_steps=1000
#     )

# Train RNN on collected latents
training.RNN_Train(epochs=20)