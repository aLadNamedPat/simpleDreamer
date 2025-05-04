"""
Complete this implementation after we've finished one training run
"""

# import gymnasium as gym
# import numpy as np

# class EnvironmentWrapper(gym.Env):
#     def __init__(
#         self,
#         vae,
#         MDN_RNN,
#         controller_action_dim
#     ):
#         """
#         ---- Idea Courtesy of GPT ----
#         Maybe we are at AGI...

#         We can redefine our environment such that it's wrapped in a gym environment since we will be making forward latent predictions
#         that the controller will use for the next action prediction. We can essentially wrap the environment such that the observations
#         that are available to the controller are from the MDNRNN and VAE

#         ------------------------------
#         """

#         self.vae = vae
#         self.rnn = MDN_RNN

#         obs_dim = vae.latent_dim + MDN_RNN.hidden_size
#         self.observation_space = gym.spaces.box(-np.inf, np.inf, (obs_dim,))

#     def reset(self):
#         init_obs = self.reset()
#         latent_var = self.vae.get_latent(init_obs)
#         hiddens = self.rnn.get_initial_hidden()
#         self.state = np.concatenate([latent_var, hiddens])
#         return self.state
    
#     def step(
#         self,
#         action
#     ):
#         pass