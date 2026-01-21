# test_rnn_synthetic.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sine_data(num_sequences=1000, seq_len=100, latent_dim=8, action_dim=2):
    """
    Generate clear sine wave sequences where action controls frequency and amplitude.
    """
    data = []
    
    for _ in range(num_sequences):
        z = np.zeros((seq_len, latent_dim), dtype=np.float32)
        a = np.zeros((seq_len, action_dim), dtype=np.float32)
        
        amplitude = np.random.uniform(0.5, 2.0)
        frequency = np.random.uniform(0.05, 0.2)
        
        a[:, 0] = amplitude
        a[:, 1] = frequency
        
        for d in range(latent_dim):
            phase = d * np.pi / latent_dim
            for t in range(seq_len):
                z[t, d] = amplitude * np.sin(frequency * t + phase)
        
        data.append({'z': z, 'a': a})
    
    return data


# Generate data
print("Generating synthetic data...")
train_data = generate_sine_data(num_sequences=500, seq_len=100, latent_dim=8, action_dim=2)
test_data = generate_sine_data(num_sequences=50, seq_len=100, latent_dim=8, action_dim=2)

# Create model
model = RNN_MDN(
    input_size=8,
    action_dim=2,
    hidden_size=128,
    num_gaussians=6,
    hidden_layer=128,
    num_layers=2
).to(device)

# Optimizer with initial learning rate
initial_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate schedulers - pick one:

# Option 1: StepLR - reduce by factor every N epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Option 2: ExponentialLR - smooth decay
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# Option 3: CosineAnnealingLR - smooth cosine decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

# Option 4: ReduceLROnPlateau - reduce when loss plateaus
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop
print("\nTraining with LR annealing...")
losses = []
lrs = []
num_epochs = 200

for epoch in range(num_epochs):
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
    
    # Track learning rate
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    
    # Step scheduler (for ReduceLROnPlateau, use scheduler.step(avg_loss))
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

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
        
        z_next = (mu * pi).sum(dim=2).squeeze(1)
        z_pred.append(z_next)
    
    z_pred = torch.cat(z_pred, dim=0).cpu().numpy()

# Also test with teacher forcing
print("Testing with teacher forcing...")
with torch.no_grad():
    z_pred_tf = []
    h = model.get_initial_hidden(device, batch_size=1)
    
    for t in range(len(test_seq['z']) - 1):
        z_t = z_true[:, t, :]  # Use TRUE z
        a_t = a_true[:, t, :]
        
        (mu, var, pi), h = model.forward(z_t, h, a_t)
        
        z_next = (mu * pi).sum(dim=2).squeeze(1)
        z_pred_tf.append(z_next.squeeze(0).cpu().numpy())
    
    z_pred_tf = np.array(z_pred_tf)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Row 1: Training metrics
axes[0, 0].plot(losses)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')

axes[0, 1].plot(lrs)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Learning Rate Schedule')
axes[0, 1].set_yscale('log')

axes[0, 2].plot(losses[-50:])
axes[0, 2].set_xlabel('Epoch (last 50)')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('Loss (Zoomed)')

# Row 2: Predictions
axes[1, 0].plot(test_seq['z'][:, 0], label='Ground Truth', linewidth=2)
axes[1, 0].plot(z_pred[:, 0], '--', label='Autoregressive', linewidth=2)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('z[0]')
axes[1, 0].set_title('Autoregressive Prediction (dim 0)')
axes[1, 0].legend()

axes[1, 1].plot(test_seq['z'][1:, 0], label='Ground Truth', linewidth=2)
axes[1, 1].plot(z_pred_tf[:, 0], '--', label='Teacher Forcing', linewidth=2)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('z[0]')
axes[1, 1].set_title('Teacher Forcing Prediction (dim 0)')
axes[1, 1].legend()

# Error comparison
error_autoreg = np.abs(z_pred[1:, 0] - test_seq['z'][1:, 0])
error_tf = np.abs(z_pred_tf[:, 0] - test_seq['z'][1:, 0])
axes[1, 2].plot(error_autoreg, label='Autoregressive', alpha=0.7)
axes[1, 2].plot(error_tf, label='Teacher Forcing', alpha=0.7)
axes[1, 2].set_xlabel('Time')
axes[1, 2].set_ylabel('Absolute Error')
axes[1, 2].set_title('Prediction Error')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('rnn_mdn_test_annealing.png')
plt.show()

print("\nSaved rnn_mdn_test_annealing.png")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"Final LR: {lrs[-1]:.6f}")
print(f"Mean autoregressive error: {error_autoreg.mean():.4f}")
print(f"Mean teacher forcing error: {error_tf.mean():.4f}")