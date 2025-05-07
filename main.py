from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v2")
training = Train(Env)
training.initialize(3, 3, 32, [64, 64, 128, 128], 3, 35, 5, hidden_layer=256, path_to_VAE_weights="vae_weights_epoch_02.pth", path_to_RNN_weights="weights/RNN_weights_epoch_256.pth")

# for i in range(1000):
#     print(i)
#     training.rollout(random_action=True, RNN_latents=True, save_root= "rollouts_2", max_steps = 1000)

# training.RNN_Train(256, 64)