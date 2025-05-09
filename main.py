from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v2")
training = Train(Env)
training.initialize(3, 3, 32, [64, 64, 128, 128], 3, 35, 5, hidden_layer=256)

# for i in range(1000):
#     print(i)
#     training.rollout(random_action=True, save_images=True, save_root= "rollouts", max_steps = 1000)

training.VAE_Train(5)