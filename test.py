import torch
from VAE import VAE
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)

vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))

print("VAE and RNN loaded successfully!")

# Quick shape test
z = torch.randn(1, 32).to(device)
a = torch.randn(1, 3).to(device)
h = rnn.get_initial_hidden(device, 1)

(mu, var, pi), h_new = rnn.forward(z, h, a)
print(f"mu: {mu.shape}, var: {var.shape}, pi: {pi.shape}")