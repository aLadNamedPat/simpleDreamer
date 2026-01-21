from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v3")
training = Train(Env)

# Initialize with random weights (no pre-trained VAE needed for image collection)
training.initialize(
    3, 3, 32, [64, 64, 128, 128], 3, 35, 5,
    hidden_layer=256,
    path_to_VAE_weights=None,  # No weights needed - just collecting images
    path_to_RNN_weights=None
)

# Collect images for VAE training
for i in range(5000):
    print(f"Image rollout {i}")
    training.rollout(
        random_action=True,
        save_images=True,       # Save raw images for VAE
        RNN_latents=False,      # Don't need latents yet
        save_root="rollouts",   # Save to rollouts directory
        max_steps=1000
    )

# Train VAE on collected images
training.VAE_Train(epochs=20)