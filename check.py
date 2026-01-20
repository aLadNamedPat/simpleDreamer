import torch
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)

vae.load_state_dict(torch.load("vae_weights_epoch_05.pth", map_location=device))
rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_20.pth", map_location=device))

# Test forward pass
z = torch.randn(1, 32).to(device)
a = torch.randn(1, 3).to(device)
h = rnn.get_initial_hidden(device, 1)

(mu, var, pi), h_new = rnn.forward(z, h, a)
print(f"RNN output shapes: mu={mu.shape}, var={var.shape}, pi={pi.shape}")

# Test controller input size
controller = Controller(32 + 35, 3).to(device)
h_for_ctrl = h_new[0][-1]  # [1, 35]
ctrl_input = torch.cat([z, h_for_ctrl], dim=1)  # [1, 67]
action = controller(ctrl_input)
print(f"Controller input: {ctrl_input.shape}, output: {action.shape}")

print("\nAll checks passed!")