"""
World Models Training Pipeline
Main entry point for data collection, VAE training, and RNN-MDN training.
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
from Train import Train


def parse_args():
    parser = argparse.ArgumentParser(description='World Models Training Pipeline')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['collect', 'train_vae', 'encode', 'train_rnn', 'full'],
                        help='Training mode: collect data, train VAE, encode rollouts, train RNN, or full pipeline')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='CarRacing-v3',
                        help='Gym environment name')
    
    # Data collection parameters
    parser.add_argument('--num_rollouts', type=int, default=5000,
                        help='Number of rollouts to collect')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per rollout')
    parser.add_argument('--data_root', type=str, default='unified_rollouts',
                        help='Root directory for saving/loading rollout data')
    parser.add_argument('--brownian_dt', type=float, default=0.1,
                        help='Brownian motion time step')
    parser.add_argument('--volatility', type=float, nargs=3, default=[0.3, 0.2, 0.1],
                        help='Brownian volatility for [steering, gas, brake]')
    parser.add_argument('--gas_bias', type=float, default=0.2,
                        help='Forward bias for gas pedal')
    parser.add_argument('--no_forward_bias', action='store_true',
                        help='Disable forward bias in action sampling')
    
    # VAE architecture
    parser.add_argument('--vae_latent_dim', type=int, default=32,
                        help='VAE latent dimension')
    parser.add_argument('--vae_hidden_dims', type=int, nargs='+', default=[64, 64, 128, 128],
                        help='VAE hidden layer dimensions')
    
    # VAE training parameters
    parser.add_argument('--vae_epochs', type=int, default=10,
                        help='Number of VAE training epochs')
    parser.add_argument('--vae_batch_size', type=int, default=64,
                        help='VAE training batch size')
    parser.add_argument('--vae_lr', type=float, default=0.0005,
                        help='VAE learning rate')
    parser.add_argument('--kld_weight', type=float, default=0.00025,
                        help='KL divergence weight for VAE loss')
    parser.add_argument('--vae_weights', type=str, default=None,
                        help='Path to pretrained VAE weights')
    
    # Encoding parameters
    parser.add_argument('--encode_batch_size', type=int, default=64,
                        help='Batch size for encoding rollouts')
    
    # RNN-MDN architecture
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action space dimension')
    parser.add_argument('--rnn_hidden_size', type=int, default=256,
                        help='RNN hidden state size')
    parser.add_argument('--num_gaussians', type=int, default=5,
                        help='Number of Gaussian components in MDN')
    parser.add_argument('--rnn_hidden_layer', type=int, default=256,
                        help='RNN-MDN hidden layer size')
    parser.add_argument('--rnn_num_layers', type=int, default=1,
                        help='Number of RNN layers')
    
    # RNN training parameters
    parser.add_argument('--rnn_epochs', type=int, default=20,
                        help='Number of RNN training epochs')
    parser.add_argument('--rnn_batch_size', type=int, default=16,
                        help='RNN training batch size')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='Sequence length for RNN training')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride for sequence sampling')
    parser.add_argument('--test_count', type=int, default=100,
                        help='Number of rollouts for test set')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate on test set every N epochs')
    parser.add_argument('--rnn_weights', type=str, default=None,
                        help='Path to pretrained RNN weights')
    
    # Wandb settings
    parser.add_argument('--vae_project', type=str, default='WorldModels_VAE',
                        help='Wandb project name for VAE training')
    parser.add_argument('--rnn_project', type=str, default='WorldModels_RNN',
                        help='Wandb project name for RNN training')
    
    return parser.parse_args()


def collect_data(trainer, args):
    """Step 1: Collect rollout data"""
    print("\n" + "=" * 70)
    print("STEP 1: COLLECTING ROLLOUTS")
    print("=" * 70)
    
    trainer.collect_rollouts(
        num_rollouts=args.num_rollouts,
        save_root=args.data_root,
        max_steps=args.max_steps,
        dt=args.brownian_dt,
        volatility=np.array(args.volatility),
        use_forward_bias=not args.no_forward_bias,
        gas_bias=args.gas_bias,
    )


def train_vae(trainer, args):
    """Step 2: Train VAE on collected images"""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING VAE")
    print("=" * 70)
    
    trainer.VAE_Train(
        epochs=args.vae_epochs,
        batch_size=args.vae_batch_size,
        kld_weight=args.kld_weight,
        learning_rate=args.vae_lr,
        data_root=args.data_root,
        project_name=args.vae_project,
    )


def encode_rollouts(trainer, args):
    """Step 3: Encode rollouts to latent representations"""
    print("\n" + "=" * 70)
    print("STEP 3: ENCODING ROLLOUTS TO LATENTS")
    print("=" * 70)
    
    trainer.encode_rollouts(
        data_root=args.data_root,
        batch_size=args.encode_batch_size,
    )


def train_rnn(trainer, args):
    """Step 4: Train RNN-MDN on encoded sequences"""
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING RNN-MDN")
    print("=" * 70)
    
    trainer.RNN_Train(
        epochs=args.rnn_epochs,
        batch_size=args.rnn_batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        test_count=args.test_count,
        eval_every=args.eval_every,
        data_root=args.data_root,
        project_name=args.rnn_project,
    )


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("WORLD MODELS TRAINING PIPELINE")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.env}")
    print(f"Data root: {args.data_root}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    
    # Create environment
    env = gym.make(args.env, continuous=True)
    
    # Initialize trainer
    trainer = Train(env)
    trainer.initialize(
        VAE_input_channels=3,
        VAE_out_channels=3,
        VAE_latent_dim=args.vae_latent_dim,
        VAE_hidden_dims=args.vae_hidden_dims,
        action_dim=args.action_dim,
        hidden_size=args.rnn_hidden_size,
        num_gaussians=args.num_gaussians,
        hidden_layer=args.rnn_hidden_layer,
        num_layers=args.rnn_num_layers,
        path_to_VAE_weights=args.vae_weights,
        path_to_RNN_weights=args.rnn_weights,
    )
    
    # Execute based on mode
    if args.mode == 'collect':
        collect_data(trainer, args)
    
    elif args.mode == 'train_vae':
        train_vae(trainer, args)
    
    elif args.mode == 'encode':
        encode_rollouts(trainer, args)
    
    elif args.mode == 'train_rnn':
        train_rnn(trainer, args)
    
    elif args.mode == 'full':
        # Run complete pipeline
        collect_data(trainer, args)
        train_vae(trainer, args)
        encode_rollouts(trainer, args)
        train_rnn(trainer, args)
    
    env.close()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()