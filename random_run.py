import gymnasium as gym
from train import Train
from Controller import Controller

Env = gym.make("CarRacing-v2", render_mode="human")
training = Train(Env)
training.initialize(3, 3, 32, [64, 64, 128, 128], 3, 35, 5, hidden_layer=256, path_to_VAE_weights="vae_weights_epoch_02.pth")

# 2. Make sure your controller knows the gym action space:
controller = Controller(
    input_features=20,      # dummy here, only random_action() uses action_space
    actions_dims=3,
    action_space=Env.action_space
)

# 3. Reset the env (returns obs, info)
obs, info = Env.reset()

for step in range(10000):
    # 4. Sample a random action from your controller
    action = controller.random_action()
    # If it’s a torch.Tensor, convert to numpy:
    if hasattr(action, "numpy"):
        action = action.numpy().squeeze()
    
    # 5. Step & render
    obs, reward, terminated, truncated, info = Env.step(action)
    
    # 6. If episode ended, reset
    if terminated or truncated:
        obs, info = Env.reset()

# 7. Close when done
Env.close()