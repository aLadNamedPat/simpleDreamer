# test_rnn_synthetic.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sine_data(num_sequences=1000, seq_len=100, latent_dim=8, action_dim=2):
    """
    Generate clear sine wave sequences where action controls frequency and amplitude.
    
    z[t, d] = amplitude * sin(frequency * t + phase)
    
    Where:
    - amplitude is controlled by action[0]
    - frequency is controlled by action[1]
    - Each latent dimension has a different phase offset
    """
    data = []
    
    for _ in range(num_sequences):
        z = np.zeros((seq_len, latent_dim), dtype=np.float32)
        a = np.zeros((seq_len, action_dim), dtype=np.float32)
        
        # Sample action once per sequence (controls the whole trajectory)
        amplitude = np.random.uniform(0.5, 2.0)  # Controls wave height
        frequency = np.random.uniform(0.05, 0.2)  # Controls wave speed
        
        # Action encodes these parameters
        a[:, 0] = amplitude
        a[:, 1] = frequency
        
        # Generate sine wave for each latent dimension with different phase
        for d in range(latent_dim):
            phase = d * np.pi / latent_dim  # Different phase per dimension
            for t in range(seq_len):
                z[t, d] = amplitude * np.sin(frequency * t + phase)
        
        data.append({'z': z, 'a': a})
    
    return data


def generate_action_dependent_dynamics(num_sequences=1000, seq_len=100, latent_dim=8, action_dim=2):
    """
    Alternative: Action at each step affects next state.
    
    z[t+1] = z[t] + action[t, 0] * velocity
    velocity += action[t, 1] * 0.1
    
    This is more like what the World Model RNN needs to learn.
    """
    data = []
    
    for _ in range(num_sequences):
        z = np.zeros((seq_len, latent_dim), dtype=np.float32)
        a = np.random.uniform(-1, 1, (seq_len, action_dim)).astype(np.float32)
        
        # Initial state
        z[0] = np.random.randn(latent_dim) * 0.1
        velocity = np.zeros(latent_dim, dtype=np.float32)
        
        for t in range(1, seq_len):
            # Action affects velocity
            velocity += a[t-1, 1] * 0.05
            velocity *= 0.95  # Damping
            
            # Action affects position change
            z[t] = z[t-1] + a[t-1, 0] * 0.1 + velocity * 0.1
        
        data.append({'z': z, 'a': a})
    
    return data


# Generate data
print("Generating synthetic data...")
train_data = generate_sine_data(num_sequences=500, seq_len=100, latent_dim=8, action_dim=2)
test_data = generate_sine_data(num_sequences=50, seq_len=100, latent_dim=8, action_dim=2)

# Visualize what the data looks like
print("Visualizing training data...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for i in range(3):
    plt.plot(train_data[i]['z'][:, 0], alpha=0.7, label=f'Seq {i}')
plt.xlabel('Time')
plt.ylabel('z[0]')
plt.title('Sample Training Sequences (dim 0)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_data[0]['z'][:, 0], label='dim 0')
plt.plot(train_data[0]['z'][:, 1], label='dim 1')
plt.plot(train_data[0]['z'][:, 2], label='dim 2')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('Single Sequence (multiple dims)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_data[0]['a'][:, 0], label='action[0] (amplitude)')
plt.plot(train_data[0]['a'][:, 1], label='action[1] (frequency)')
plt.xlabel('Time')
plt.ylabel('Action')
plt.title('Actions for Sequence 0')
plt.legend()

plt.tight_layout()
plt.savefig('data_visualization.png')
print("Saved data_visualization.png")

# Create model
model = RNN_MDN(
    input_size=8,
    action_dim=2,
    hidden_size=64,
    num_gaussians=3,
    hidden_layer=32,
    num_layers=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining...")
losses = []

for epoch in range(50):
    epoch_loss = 0
    np.random.shuffle(train_data)
    
    for seq in train_data:
        z = torch.from_numpy(seq['z']).unsqueeze(0).to(device)
        a = torch.from_numpy(seq['a']).unsqueeze(0).to(device)
        
        x = torch.cat([z[:, :-1, :], a[:, :-1, :]], dim=-1)
        y = z[:, 1:, :]
        
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
        
        # Weighted average of mixture means
        z_next = (mu * pi).sum(dim=2).squeeze(1)
        z_pred.append(z_next)
    
    z_pred = torch.cat(z_pred, dim=0).cpu().numpy()

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(test_seq['z'][:, 0], label='Ground Truth', linewidth=2)
plt.plot(z_pred[:, 0], '--', label='Predicted', linewidth=2)
plt.xlabel('Time')
plt.ylabel('z[0]')
plt.title('Prediction vs Ground Truth (dim 0)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_seq['z'][:, 1], label='Ground Truth', linewidth=2)
plt.plot(z_pred[:, 1], '--', label='Predicted', linewidth=2)
plt.xlabel('Time')
plt.ylabel('z[1]')
plt.title('Prediction vs Ground Truth (dim 1)')
plt.legend()

plt.tight_layout()
plt.savefig('rnn_mdn_test.png')
plt.show()

print("\nSaved rnn_mdn_test.png")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")