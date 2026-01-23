# train_controller.py

import torch
import numpy as np
import gymnasium as gym
from VAE import VAE
from RNN_MDN import RNN_MDN
from Controller import Controller
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_controller(vae, rnn, controller, env, max_steps=1000):
    """Evaluate controller in real environment. Returns total reward."""
    vae.eval()
    rnn.eval()
    controller.eval()
    
    with torch.no_grad():
        obs, _ = env.reset()
        h = rnn.get_initial_hidden(device, batch_size=1)
        total_reward = 0
        
        for step in range(max_steps):
            # Encode observation
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparamterize(mu, logvar)
            
            # Get action from controller
            h_for_controller = h[0][-1]  # [1, hidden_size]
            controller_input = torch.cat([z, h_for_controller], dim=1)
            
            action = controller(controller_input)
            action = torch.tanh(action)
            
            # Scale to CarRacing action space
            a = action.squeeze(0).cpu().numpy()
            a[1] = (a[1] + 1) / 2  # gas: [-1,1] -> [0,1]
            a[2] = (a[2] + 1) / 2  # brake: [-1,1] -> [0,1]
            a = a.astype(np.float32)
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(a)
            total_reward += reward
            
            # Update RNN hidden state
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            (_, _, _), h = rnn.forward(z, h, a_tensor)
            
            if done or truncated:
                break
        
        return total_reward


def get_controller_params(controller):
    """Flatten all controller parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in controller.parameters()])


def set_controller_params(controller, params):
    """Set controller parameters from a flattened vector."""
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data.copy_(params[idx:idx+size].view(p.shape))
        idx += size


def train_controller_es(
    vae, 
    rnn, 
    controller, 
    env,
    generations=100,
    population_size=32,
    sigma=0.1,
    learning_rate=0.01,
    eval_episodes=3
):
    """
    Train controller using simple Evolution Strategy.
    """
    vae.eval()
    rnn.eval()
    
    # Get initial parameters
    params = get_controller_params(controller)
    num_params = len(params)
    print(f"Training controller with {num_params} parameters")
    
    best_reward = -float('inf')
    best_params = params.clone()
    
    for gen in range(generations):
        # Generate population of perturbations
        noise = torch.randn(population_size, num_params, device=device)
        print(f"Current Generation: {gen}")
        rewards = []
        
        for i in range(population_size):
            # Create perturbed controller
            perturbed_params = params + sigma * noise[i]
            print(f"Current population: {i}")

            set_controller_params(controller, perturbed_params)

            # Evaluate over multiple episodes
            ep_rewards = []
            for _ in range(eval_episodes):
                r = evaluate_controller(vae, rnn, controller, env)
                ep_rewards.append(r)
            print(f"Obtained reward {r}")
            avg_reward = np.mean(ep_rewards)
            rewards.append(avg_reward)
        
        rewards = np.array(rewards)
        
        # Normalize rewards
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Update parameters (gradient estimate)
        grad = torch.zeros(num_params, device=device)
        for i in range(population_size):
            grad += rewards_norm[i] * noise[i]
        grad /= (population_size * sigma)
        
        params = params + learning_rate * grad
        
        # Evaluate current best
        set_controller_params(controller, params)
        current_reward = np.mean([evaluate_controller(vae, rnn, controller, env) for _ in range(3)])
        
        if current_reward > best_reward:
            best_reward = current_reward
            best_params = params.clone()
        
        print(f"Gen {gen+1:3d} | Mean: {rewards.mean():7.1f} | Max: {rewards.max():7.1f} | Best: {best_reward:7.1f}")
        
        # Save checkpoint every 10 generations
        if (gen + 1) % 10 == 0:
            set_controller_params(controller, best_params)
            torch.save(controller.state_dict(), f"controller_gen_{gen+1:03d}.pth")
    
    # Restore best parameters
    set_controller_params(controller, best_params)
    return best_reward


def main():
    env = gym.make("CarRacing-v3")
    
    # Load VAE and RNN
    vae = VAE(3, 3, 32, [64, 64, 128, 128]).to(device)
    rnn = RNN_MDN(32, 3, 35, 5, 256, 1).to(device)
    
    vae.load_state_dict(torch.load("vae_weights_epoch_04.pth", map_location=device))
    rnn.load_state_dict(torch.load("weights/RNN_weights_epoch_50.pth", map_location=device))
    
    # Initialize controller
    # Input: z (32) + h (35) = 67
    controller = Controller(
        input_features=32 + 35,
        actions_dims=3,
    ).to(device)
    
    print("Models loaded!")
    
    # Baseline
    print("\n=== Baseline ===")
    baseline = np.mean([evaluate_controller(vae, rnn, controller, env) for _ in range(3)])
    print(f"Untrained controller: {baseline:.1f}")
    
    # Train
    print("\n=== Training Controller ===")
    best_reward = train_controller_es(
        vae, rnn, controller, env,
        generations=100,
        population_size=16,  # Smaller for faster iteration
        sigma=0.1,
        learning_rate=0.03,
        eval_episodes=1  # 1 episode per candidate for speed
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_rewards = [evaluate_controller(vae, rnn, controller, env) for _ in range(10)]
    print(f"Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    
    # Save final controller
    torch.save(controller.state_dict(), "controller_final.pth")
    print("Saved controller_final.pth")
    
    env.close()


if __name__ == "__main__":
    main()