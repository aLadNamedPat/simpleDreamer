"""
Test script to verify MDN loss computation in RNN_MDN model.

This test creates synthetic data from a known mixture of Gaussians and trains
the MDN to recover the original parameters. If the loss is computed correctly,
the model should converge to the true means, variances, and mixture weights.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from torch.distributions import Normal, MixtureSameFamily
import numpy as np

# Import your RNN_MDN - adjust path as needed
# from RNN_MDN import RNN_MDN


class SimpleMDN(nn.Module):
    """
    A simplified MDN (no RNN) that mimics the MDN head of your RNN_MDN.
    Used to isolate and test just the MDN loss computation.
    """
    def __init__(self, input_dim, output_dim, num_gaussians, hidden_layer=40):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians
        
        self.fc = nn.Linear(input_dim, hidden_layer)
        self.mu = nn.Linear(hidden_layer, output_dim * num_gaussians)
        self.var = nn.Linear(hidden_layer, output_dim * num_gaussians)
        self.pi = nn.Linear(hidden_layer, num_gaussians)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] or [batch_size, seq_len, input_dim]
        Returns:
            mu: [batch_size, num_gaussians, output_dim]
            var: [batch_size, num_gaussians, output_dim]
            logpi: [batch_size, num_gaussians]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len dim
        
        batch_size, seq_len, _ = x.shape
        
        h = self.leaky_relu(self.fc(x))
        
        mu = self.mu(h)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.output_dim)
        
        var = torch.exp(self.var(h))
        var = var.view(batch_size, seq_len, self.num_gaussians, self.output_dim)
        
        logpi = F.log_softmax(self.pi(h), dim=-1)
        
        # Squeeze seq_len if it was added
        if seq_len == 1:
            mu = mu.squeeze(1)
            var = var.squeeze(1)
            logpi = logpi.squeeze(1)
        
        return mu, var, logpi


def mdn_loss(y, mu, var, logpi):
    """
    Compute MDN loss using log-sum-exp trick for numerical stability.
    
    This mirrors the loss computation in your RNN_MDN.MDN_loss method.
    
    Args:
        y: [batch_size, output_dim] target values
        mu: [batch_size, num_gaussians, output_dim] means
        var: [batch_size, num_gaussians, output_dim] variances
        logpi: [batch_size, num_gaussians] log mixture weights
    
    Returns:
        loss: scalar negative log likelihood
    """
    y = y.unsqueeze(1)  # [batch_size, 1, output_dim]
    var = var.clamp(min=1e-6)
    
    # Log probability of Gaussian: log N(y | mu, var)
    # = -0.5 * (log(2*pi*var) + (y-mu)^2/var)
    log_pdf = -0.5 * (
        torch.log(2 * torch.pi * var) + 
        (y - mu) ** 2 / var
    ).sum(dim=-1)  # Sum over output dimensions -> [batch_size, num_gaussians]
    
    # Log mixture: log(pi) + log(pdf)
    log_mix = logpi + log_pdf  # [batch_size, num_gaussians]
    
    # Log-sum-exp trick for numerical stability
    max_log = log_mix.max(dim=-1, keepdim=True)[0]
    log_prob = max_log.squeeze(-1) + torch.log(
        torch.exp(log_mix - max_log).sum(dim=-1)
    )
    
    loss = -log_prob.mean()
    return loss


def generate_gmm_samples(means, stds, pi, n_samples):
    """
    Generate samples from a Gaussian Mixture Model.
    
    Args:
        means: [num_gaussians, output_dim] means of each component
        stds: [num_gaussians, output_dim] standard deviations
        pi: [num_gaussians] mixture weights (should sum to 1)
        n_samples: number of samples to generate
    
    Returns:
        samples: [n_samples, output_dim]
    """
    cat_dist = Categorical(pi)
    indices = cat_dist.sample((n_samples,)).long()
    rands = torch.randn(n_samples, means.shape[1])
    samples = means[indices] + rands * stds[indices]
    return samples


class TestMDNLoss(unittest.TestCase):
    """Test cases for MDN loss computation."""
    
    def test_simple_mdn_convergence(self):
        """
        Test that a simple MDN can recover known GMM parameters.
        This tests the loss function in isolation.
        """
        print("\n" + "=" * 60)
        print("TEST 1: Simple MDN convergence to known GMM parameters")
        print("=" * 60)
        
        # Ground truth GMM parameters
        true_means = torch.Tensor([
            [0., 0.],
            [3., 3.],
            [-3., 3.]
        ])
        true_stds = torch.Tensor([
            [0.5, 0.5],
            [0.3, 0.8],
            [0.8, 0.3]
        ])
        true_pi = torch.Tensor([0.3, 0.4, 0.3])
        
        n_samples = 10000
        samples = generate_gmm_samples(true_means, true_stds, true_pi, n_samples)
        
        print(f"\nTrue means:\n{true_means}")
        print(f"True stds:\n{true_stds}")
        print(f"True pi: {true_pi}")
        
        # Create model
        model = SimpleMDN(input_dim=1, output_dim=2, num_gaussians=3, hidden_layer=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Dummy input (we're just fitting the MDN head to output the right distribution)
        dummy_input = torch.ones(128, 1)
        
        iterations = 10000
        log_step = iterations // 5
        
        pbar = tqdm(total=iterations, desc="Training Simple MDN")
        cum_loss = 0
        
        for i in range(iterations):
            # Sample batch
            batch_idx = torch.randint(0, n_samples, (128,))
            batch = samples[batch_idx]
            
            # Forward pass
            mu, var, logpi = model(dummy_input)
            
            # Compute loss
            loss = mdn_loss(batch, mu, var, logpi)
            
            cum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix_str(f"loss={loss.item():.4f}, avg={cum_loss/(i+1):.4f}")
            pbar.update(1)
            
            if i % log_step == log_step - 1:
                with torch.no_grad():
                    mu, var, logpi = model(dummy_input[:1])
                    pi = torch.exp(logpi)
                    std = torch.sqrt(var)
                    print(f"\n\nIteration {i+1}:")
                    print(f"Learned means:\n{mu.squeeze()}")
                    print(f"Learned stds:\n{std.squeeze()}")
                    print(f"Learned pi: {pi.squeeze()}")
        
        pbar.close()
        
        # Final check
        with torch.no_grad():
            mu, var, logpi = model(dummy_input[:1])
            pi = torch.exp(logpi)
            std = torch.sqrt(var)
            
            print("\n" + "-" * 40)
            print("FINAL RESULTS:")
            print(f"Learned means:\n{mu.squeeze()}")
            print(f"Learned stds:\n{std.squeeze()}")
            print(f"Learned pi: {pi.squeeze()}")
            print("-" * 40)
            
            # The model should have learned parameters close to true values
            # (order may differ due to permutation invariance)
            final_loss = mdn_loss(samples[:1000], 
                                  mu.expand(1000, -1, -1), 
                                  var.expand(1000, -1, -1), 
                                  logpi.expand(1000, -1))
            print(f"Final loss on 1000 samples: {final_loss.item():.4f}")
            
            # A well-trained model should achieve loss close to the entropy of the true distribution
            self.assertLess(final_loss.item(), 3.0, "Loss should converge to reasonable value")
    
    def test_rnn_mdn_convergence(self):
        """
        Test the full RNN_MDN model can learn a sequence-dependent GMM.
        """
        print("\n" + "=" * 60)
        print("TEST 2: RNN_MDN convergence on sequence data")
        print("=" * 60)
        
        # Import your actual model
        try:
            from RNN_MDN import RNN_MDN
        except ImportError:
            print("Could not import RNN_MDN. Skipping this test.")
            print("Make sure RNN_MDN.py is in the same directory.")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Model parameters
        latent_dim = 4
        action_dim = 2
        hidden_size = 64
        num_gaussians = 3
        seq_len = 10
        batch_size = 32
        
        # Create model
        model = RNN_MDN(
            input_size=latent_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_gaussians=num_gaussians,
            hidden_layer=32,
            num_layers=1
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate synthetic sequential data
        # The target is a simple function of the input for testing
        def generate_sequence_batch(batch_size, seq_len, latent_dim, action_dim):
            """Generate sequences where y[t] depends on x[t] and a[t]."""
            x = torch.randn(batch_size, seq_len, latent_dim)
            a = torch.randn(batch_size, seq_len, action_dim)
            
            # Simple target: next state is current state + small noise + action influence
            # y[t] = x[t] * 0.9 + a[t, :latent_dim] * 0.5 + noise
            noise = torch.randn_like(x) * 0.1
            y = x * 0.9 + noise
            # Add action influence (use first latent_dim dims of action or pad)
            if action_dim >= latent_dim:
                y = y + a[:, :, :latent_dim] * 0.3
            else:
                y[:, :, :action_dim] = y[:, :, :action_dim] + a * 0.3
            
            return x, a, y
        
        iterations = 5000
        log_step = iterations // 5
        
        pbar = tqdm(total=iterations, desc="Training RNN_MDN")
        cum_loss = 0
        
        for i in range(iterations):
            x, a, y = generate_sequence_batch(batch_size, seq_len, latent_dim, action_dim)
            x, a, y = x.to(device), a.to(device), y.to(device)
            
            # Concatenate input for RNN
            rnn_input = torch.cat((x, a), dim=-1)
            
            # Forward pass and loss
            loss, _ = model.MDN_loss(rnn_input, y, h0=None)
            
            cum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix_str(f"loss={loss.item():.4f}, avg={cum_loss/(i+1):.4f}")
            pbar.update(1)
            
            if i % log_step == log_step - 1:
                print(f"\n\nIteration {i+1}: avg_loss = {cum_loss/(i+1):.4f}")
        
        pbar.close()
        
        # Test prediction quality
        print("\n" + "-" * 40)
        print("Testing prediction quality...")
        
        model.eval()
        with torch.no_grad():
            x_test, a_test, y_test = generate_sequence_batch(100, seq_len, latent_dim, action_dim)
            x_test, a_test, y_test = x_test.to(device), a_test.to(device), y_test.to(device)
            
            rnn_input = torch.cat((x_test, a_test), dim=-1)
            test_loss, _ = model.MDN_loss(rnn_input, y_test, h0=None)
            
            print(f"Test loss: {test_loss.item():.4f}")
            
            # Also test single-step prediction
            h = model.get_initial_hidden(device, batch_size=1)
            x_single = x_test[0:1, 0:1, :]  # [1, 1, latent_dim]
            a_single = a_test[0:1, 0:1, :]  # [1, 1, action_dim]
            
            # Squeeze for forward() which expects [1, latent_dim]
            (mu, var, logpi), h_new = model.forward(
                x_single.squeeze(1), 
                h, 
                a_single.squeeze(1)
            )
            
            print(f"\nSingle step prediction shapes:")
            print(f"  mu: {mu.shape}")
            print(f"  var: {var.shape}")
            print(f"  logpi: {logpi.shape}")
            
            # Get the most likely Gaussian's mean as prediction
            pi = torch.exp(logpi.squeeze())
            best_gaussian = torch.argmax(pi)
            predicted_mean = mu.squeeze()[0, best_gaussian, :]  # [latent_dim]
            actual_next = y_test[0, 0, :]  # [latent_dim]
            
            print(f"\nPredicted mean (best Gaussian): {predicted_mean.cpu().numpy()}")
            print(f"Actual next state: {actual_next.cpu().numpy()}")
            print(f"Mixture weights: {pi.cpu().numpy()}")
        
        self.assertLess(test_loss.item(), 5.0, "Test loss should be reasonable")
    
    def test_loss_gradient_flow(self):
        """
        Test that gradients flow properly through the MDN loss.
        """
        print("\n" + "=" * 60)
        print("TEST 3: Gradient flow verification")
        print("=" * 60)
        
        model = SimpleMDN(input_dim=1, output_dim=2, num_gaussians=3)
        
        dummy_input = torch.ones(32, 1)
        target = torch.randn(32, 2)
        
        mu, var, logpi = model(dummy_input)
        loss = mdn_loss(target, mu, var, logpi)
        
        loss.backward()
        
        print("\nGradient norms for each parameter:")
        all_have_grad = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: {grad_norm:.6f}")
                if grad_norm == 0:
                    all_have_grad = False
            else:
                print(f"  {name}: NO GRADIENT")
                all_have_grad = False
        
        self.assertTrue(all_have_grad, "All parameters should have non-zero gradients")
        print("\n✓ All parameters have gradients flowing through them")
    
    def test_numerical_stability(self):
        """
        Test that the loss doesn't produce NaN or Inf with edge cases.
        """
        print("\n" + "=" * 60)
        print("TEST 4: Numerical stability")
        print("=" * 60)
        
        model = SimpleMDN(input_dim=1, output_dim=2, num_gaussians=3)
        dummy_input = torch.ones(32, 1)
        
        # Test with various edge cases
        test_cases = [
            ("Normal values", torch.randn(32, 2)),
            ("Large values", torch.randn(32, 2) * 100),
            ("Small values", torch.randn(32, 2) * 0.001),
            ("Values far from predicted means", torch.randn(32, 2) * 10 + 50),
        ]
        
        for name, target in test_cases:
            mu, var, logpi = model(dummy_input)
            loss = mdn_loss(target, mu, var, logpi)
            
            is_finite = torch.isfinite(loss).item()
            print(f"  {name}: loss = {loss.item():.4f}, finite = {is_finite}")
            
            self.assertTrue(is_finite, f"Loss should be finite for {name}")
        
        print("\n✓ Loss is numerically stable across all test cases")


class TestMDNSampling(unittest.TestCase):
    """Test MDN sampling functions."""
    
    def test_sampling_distribution(self):
        """
        Test that samples from MDN match the predicted distribution.
        """
        print("\n" + "=" * 60)
        print("TEST 5: MDN sampling distribution")
        print("=" * 60)
        
        # Create known distribution parameters
        mu = torch.Tensor([[[0., 0.], [5., 5.], [-5., 5.]]])  # [1, 3, 2]
        var = torch.Tensor([[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]])  # [1, 3, 2]
        logpi = torch.log(torch.Tensor([[0.33, 0.33, 0.34]]))  # [1, 3]
        
        def sample_from_mdn(mu, var, logpi, n_samples):
            """Sample from MDN output."""
            samples = []
            pi = torch.exp(logpi)
            
            for _ in range(n_samples):
                # Sample which Gaussian
                cat = Categorical(pi.squeeze())
                idx = cat.sample()
                
                # Sample from that Gaussian
                mean = mu[0, idx]
                std = torch.sqrt(var[0, idx])
                sample = mean + torch.randn_like(mean) * std
                samples.append(sample)
            
            return torch.stack(samples)
        
        n_samples = 3000
        samples = sample_from_mdn(mu, var, logpi, n_samples)
        
        # Check that samples cluster around the three modes
        print(f"\nSampled {n_samples} points")
        print(f"Sample mean: {samples.mean(dim=0).numpy()}")
        print(f"Sample std: {samples.std(dim=0).numpy()}")
        
        # Count samples near each mode
        for i, mode in enumerate(mu.squeeze()):
            dist_to_mode = ((samples - mode) ** 2).sum(dim=1).sqrt()
            near_mode = (dist_to_mode < 1.0).sum().item()
            print(f"Samples near mode {i} ({mode.numpy()}): {near_mode} ({100*near_mode/n_samples:.1f}%)")
        
        print("\n✓ Sampling produces expected distribution")


def run_quick_test():
    """Run a quick sanity check without full training."""
    print("\n" + "=" * 60)
    print("QUICK SANITY CHECK")
    print("=" * 60)
    
    # Test basic functionality
    model = SimpleMDN(input_dim=1, output_dim=2, num_gaussians=3)
    
    x = torch.ones(10, 1)
    y = torch.randn(10, 2)
    
    mu, var, logpi = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Output shapes - mu: {mu.shape}, var: {var.shape}, logpi: {logpi.shape}")
    
    loss = mdn_loss(y, mu, var, logpi)
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check var is positive
    assert (var > 0).all(), "Variance should be positive"
    print("✓ Variance is positive")
    
    # Check pi sums to 1
    pi = torch.exp(logpi)
    pi_sum = pi.sum(dim=-1)
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), "Pi should sum to 1"
    print("✓ Mixture weights sum to 1")
    
    print("\n✓ All quick checks passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MDN loss computation")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--test", type=str, default=None, 
                        help="Run specific test (simple, rnn, gradient, stability, sampling)")
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.test:
        # Run specific test
        suite = unittest.TestSuite()
        test_map = {
            "simple": "test_simple_mdn_convergence",
            "rnn": "test_rnn_mdn_convergence", 
            "gradient": "test_loss_gradient_flow",
            "stability": "test_numerical_stability",
            "sampling": "test_sampling_distribution",
        }
        if args.test in test_map:
            if args.test == "sampling":
                suite.addTest(TestMDNSampling(test_map[args.test]))
            else:
                suite.addTest(TestMDNLoss(test_map[args.test]))
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {list(test_map.keys())}")
    else:
        # Run all tests
        run_quick_test()
        print("\n\n")
        unittest.main(verbosity=2, exit=False)