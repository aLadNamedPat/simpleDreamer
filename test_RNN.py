# test_rnn_sine_prediction.py
"""
Proper sequence prediction task: Given previous points of a sine wave, predict the next point.
No action cheating - actions are random/meaningless or can be removed entirely.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from RNN_MDN import RNN_MDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sine_sequences(num_sequences=1000, seq_len=100, latent_dim=8):
    """
    Generate sine wave sequences with varying frequencies and phases.
    The model must learn to continue the pattern from context alone.
    """
    data = []
    
    for _ in range(num_sequences):
        z = np.zeros((seq_len, latent_dim), dtype=np.float32)
        
        # Random parameters for this sequence
        amplitude = np.random.uniform(0.5, 2.0)
        frequency = np.random.uniform(0.05, 0.15)  # cycles per timestep
        phase_offset = np.random.uniform(0, 2 * np.pi)
        
        for d in range(latent_dim):
            phase = phase_offset + d * np.pi / latent_dim
            for t in range(seq_len):
                z[t, d] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        data.append({'z': z})
    
    return data


def train_model(model, train_data, num_epochs=200, use_actions=False, action_dim=2):
    """Train the model on sequence prediction."""
    
    initial_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    losses = []
    lrs = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        np.random.shuffle(train_data)
        
        for seq in train_data:
            z = torch.from_numpy(seq['z']).unsqueeze(0).to(device)  # [1, seq_len, latent_dim]
            
            if use_actions:
                # Random actions - model should learn to ignore them
                a = torch.randn(1, z.shape[1], action_dim, device=device) * 0.1
                x = torch.cat([z[:, :-1, :], a[:, :-1, :]], dim=-1)
            else:
                # No actions - pure sequence prediction
                x = z[:, :-1, :]
            
            y = z[:, 1:, :]  # Target: next timestep
            
            loss, _ = model.MDN_loss(x, y, None)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        losses.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    return losses, lrs


def predict_autoregressive(model, initial_context, num_steps, use_actions=False, action_dim=2):
    """
    Autoregressive prediction: use model's own outputs as next input.
    
    Args:
        initial_context: [context_len, latent_dim] - seed the model with this
        num_steps: how many steps to predict beyond context
    """
    model.eval()
    context_len = initial_context.shape[0]
    latent_dim = initial_context.shape[1]
    
    with torch.no_grad():
        # Initialize hidden state by running through context
        h = model.get_initial_hidden(device, batch_size=1)
        
        z_current = initial_context[0:1, :].to(device)  # [1, latent_dim]
        
        # Run through context to build up hidden state
        for t in range(context_len - 1):
            z_t = initial_context[t:t+1, :].to(device)
            if use_actions:
                a_t = torch.zeros(1, action_dim, device=device)
                (mu, var, pi), h = model.forward(z_t, h, a_t)
            else:
                # For no-action model, we need to modify forward or pass zeros
                a_t = torch.zeros(1, 0, device=device)  # empty action
                inp = z_t.unsqueeze(1)
                out, h = model.rnn(inp, h)
                mu, var, pi = model.MDN(out)
        
        # Now predict autoregressively
        z_current = initial_context[-1:, :].to(device)
        predictions = []
        
        for t in range(num_steps):
            if use_actions:
                a_t = torch.zeros(1, action_dim, device=device)
                (mu, var, pi), h = model.forward(z_current, h, a_t)
            else:
                inp = z_current.unsqueeze(1)
                out, h = model.rnn(inp, h)
                mu, var, pi = model.MDN(out)
            
            # Get prediction: weighted mean of Gaussian components
            # mu: [B, T, K, L], pi: [B, T, K]
            mu = mu.squeeze(1)  # [B, K, L]
            pi = pi.squeeze(1)  # [B, K]
            
            # Weighted average across components
            z_next = (mu * pi.unsqueeze(-1)).sum(dim=1)  # [B, L]
            
            predictions.append(z_next.cpu().numpy())
            z_current = z_next
        
    return np.array(predictions).squeeze()  # [num_steps, latent_dim]


def predict_teacher_forcing(model, z_true, use_actions=False, action_dim=2):
    """Predict with teacher forcing - always use ground truth as input."""
    model.eval()
    seq_len = z_true.shape[0]
    
    with torch.no_grad():
        h = model.get_initial_hidden(device, batch_size=1)
        predictions = []
        
        for t in range(seq_len - 1):
            z_t = z_true[t:t+1, :].to(device)
            
            if use_actions:
                a_t = torch.zeros(1, action_dim, device=device)
                (mu, var, pi), h = model.forward(z_t, h, a_t)
            else:
                inp = z_t.unsqueeze(1)
                out, h = model.rnn(inp, h)
                mu, var, pi = model.MDN(out)
            
            mu = mu.squeeze(1)
            pi = pi.squeeze(1)
            z_next = (mu * pi.unsqueeze(-1)).sum(dim=1)
            
            predictions.append(z_next.cpu().numpy())
    
    return np.array(predictions).squeeze()


def main():
    print("="*60)
    print("Sine Wave Prediction Test")
    print("Task: Given previous points, predict the next point")
    print("="*60)
    
    # Configuration
    latent_dim = 8
    use_actions = False  # Set to True to test with dummy actions
    action_dim = 2 if use_actions else 0
    
    # Generate data
    print("\nGenerating synthetic data...")
    train_data = generate_sine_sequences(num_sequences=500, seq_len=100, latent_dim=latent_dim)
    test_data = generate_sine_sequences(num_sequences=50, seq_len=100, latent_dim=latent_dim)
    
    # Create model
    input_size = latent_dim
    model = RNN_MDN(
        input_size=input_size,
        action_dim=action_dim,
        hidden_size=128,
        num_gaussians=5,
        hidden_layer=128,
        num_layers=2
    ).to(device)
    
    print(f"\nModel config:")
    print(f"  Input size: {input_size}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden size: 128")
    print(f"  Num Gaussians: 5")
    print(f"  Num layers: 2")
    
    # Train
    print("\nTraining...")
    losses, lrs = train_model(
        model, train_data, 
        num_epochs=200, 
        use_actions=use_actions, 
        action_dim=action_dim
    )
    
    # Test
    print("\nTesting...")
    test_seq = test_data[0]
    z_true = torch.from_numpy(test_seq['z'])
    
    # Use first 20 steps as context, predict remaining 80
    context_len = 20
    context = z_true[:context_len]
    
    # Autoregressive prediction
    z_pred_ar = predict_autoregressive(
        model, context, 
        num_steps=len(test_seq['z']) - context_len,
        use_actions=use_actions,
        action_dim=action_dim
    )
    
    # Teacher forcing prediction
    z_pred_tf = predict_teacher_forcing(
        model, z_true,
        use_actions=use_actions,
        action_dim=action_dim
    )
    
    # Combine context with predictions for plotting
    z_full_ar = np.concatenate([context.numpy(), z_pred_ar], axis=0)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Row 1: Training metrics
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(lrs)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(losses[-50:])
    axes[0, 2].set_xlabel('Epoch (last 50)')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Loss (Zoomed)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Predictions
    dim_to_plot = 0
    
    # Autoregressive
    axes[1, 0].plot(test_seq['z'][:, dim_to_plot], label='Ground Truth', linewidth=2)
    axes[1, 0].plot(z_full_ar[:, dim_to_plot], '--', label='Autoregressive', linewidth=2)
    axes[1, 0].axvline(x=context_len, color='r', linestyle=':', label='Context boundary')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel(f'z[{dim_to_plot}]')
    axes[1, 0].set_title(f'Autoregressive Prediction (dim {dim_to_plot})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Teacher forcing
    axes[1, 1].plot(test_seq['z'][1:, dim_to_plot], label='Ground Truth', linewidth=2)
    axes[1, 1].plot(z_pred_tf[:, dim_to_plot], '--', label='Teacher Forcing', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel(f'z[{dim_to_plot}]')
    axes[1, 1].set_title(f'Teacher Forcing Prediction (dim {dim_to_plot})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error comparison
    error_autoreg = np.abs(z_full_ar[context_len:, dim_to_plot] - test_seq['z'][context_len:, dim_to_plot])
    error_tf = np.abs(z_pred_tf[context_len-1:, dim_to_plot] - test_seq['z'][context_len:, dim_to_plot])
    
    axes[1, 2].plot(error_autoreg, label='Autoregressive', alpha=0.7)
    axes[1, 2].plot(error_tf, label='Teacher Forcing', alpha=0.7)
    axes[1, 2].set_xlabel('Time (after context)')
    axes[1, 2].set_ylabel('Absolute Error')
    axes[1, 2].set_title('Prediction Error')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_mdn_sine_prediction.png', dpi=150)
    plt.show()
    
    # Print stats
    print("\n" + "="*60)
    print("Results:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Mean autoregressive error: {error_autoreg.mean():.4f}")
    print(f"  Mean teacher forcing error: {error_tf.mean():.4f}")
    print(f"  Error ratio (AR/TF): {error_autoreg.mean() / (error_tf.mean() + 1e-8):.2f}x")
    print("="*60)
    
    # Also test on multiple dimensions
    print("\nPer-dimension autoregressive errors:")
    for d in range(min(4, latent_dim)):
        err = np.abs(z_full_ar[context_len:, d] - test_seq['z'][context_len:, d]).mean()
        print(f"  Dim {d}: {err:.4f}")


if __name__ == "__main__":
    main()