from train import *
import gymnasium as gym

Env = gym.make("CarRacing-v2")
training = Train(Env)
training.initialize(3, 3, 32, [64, 64, 128, 128], 3, 50, 26)

for i in range(1000):
    print(i)
    training.rollout(random_action=True, save_images=True, max_steps = 500)

# training.VAE_Train(10)