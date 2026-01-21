# test_rnn_synthetic.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate simple synthetic data: sine wave with action-dependent frequency
def generate_sine_data(num_sequences=1000, seq_len=100, latent_dim=8, action_dim=2):
    """
    Generate sequences where next state depends on current state + action.
    z_{t+1} = z_t + action[0] * sin(t * action[1])
    """
    data = []
    
    for _ in range(num_sequences):
        z = np.zeros((seq_len, latent_dim), dtype=np.float32)
        a = np.random.randn(seq_len, action_dim).astype(np.float32) * 0.5
        
        z[0] = np.random.randn(latent_dim) * 0.1
        
        for t in range(1, seq_len):
            # Simple dynamics: each dimension evolves based on action
            freq = 0.1 + 0.1 * a[t-1, 1]
            z[t] = z[t-1] + 0.1 * np.sin(t * freq) * a[t-1, 0] + np.random.randn(latent_dim) * 0.01
        
        data.append({'z': z, 'a': a})
    
    return data

# Generate data
print("Generating synthetic data...")
train_data = generate_sine_data(num_sequences=500, seq_len=100, latent_dim=8, action_dim=2)
test_data = generate_sine_data(num_sequences=50, seq_len=100, latent_dim=8, action_dim=2)

# Create model
model = RNN_MDN(
    input_size=8,       # latent_dim
    action_dim=2,
    hidden_size=64,
    num_gaussians=3,
    hidden_layer=32,
    num_layers=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training...")
losses = []

for epoch in range(50):
    epoch_loss = 0
    np.random.shuffle(train_data)
    
    for seq in train_data:
        z = torch.from_numpy(seq['z']).unsqueeze(0).to(device)  # [1, T, latent_dim]
        a = torch.from_numpy(seq['a']).unsqueeze(0).to(device)  # [1, T, action_dim]
        
        x = torch.cat([z[:, :-1, :], a[:, :-1, :]], dim=-1)  # Input: [1, T-1, latent+action]
        y = z[:, 1:, :]  # Target: [1, T-1, latent_dim]
        
        loss, _ = model.MDN_loss(x, y, None)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_data)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Test: Generate predictions
print("\nTesting predictions...")
model.eval()
test_seq = test_data[0]
z_true = torch.from_numpy(test_seq['z']).unsqueeze(0).to(device)
a_true = torch.from_numpy(test_seq['a']).unsqueeze(0).to(device)

with torch.no_grad():
    z_pred = [z_true[:, 0, :]]
    h = model.get_initial_hidden(device, batch_size=1)
    
    for t in range(len(test_seq['z']) - 1):
        z_t = z_pred[-1]
        a_t = a_true[:, t, :]
        
        (mu, var, pi), h = model.forward(z_t, h, a_t)
        
        # Use mean of mixture (simplified)
        z_next = (mu * pi).sum(dim=2).squeeze(1)  # Weighted average
        z_pred.append(z_next)
    
    z_pred = torch.cat(z_pred, dim=0).cpu().numpy()

# Plot predictions vs ground truth
plt.subplot(1, 2, 2)
dim_to_plot = 0
plt.plot(test_seq['z'][:, dim_to_plot], label='Ground Truth', alpha=0.7)
plt.plot(z_pred[:, dim_to_plot], label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel(f'z[{dim_to_plot}]')
plt.title('Prediction vs Ground Truth')
plt.legend()

plt.tight_layout()
plt.savefig('rnn_mdn_test.png')
plt.show()

print("Saved rnn_mdn_test.png")