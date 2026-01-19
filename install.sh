#!/bin/bash

# Installation script for World Models implementation
# This script sets up the environment for the simpleDreamer project

echo "🚗 Setting up World Models Environment..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA version if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "💻 Installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📋 Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p rollouts
mkdir -p rollouts_2
mkdir -p weights

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import gymnasium
import numpy as np
import PIL
import wandb
import cma
import matplotlib
print('✅ All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the training pipeline:"
echo "  python main.py"
echo ""
echo "To generate rollouts:"
echo "  python random_run.py"
echo ""
echo "Note: Make sure to login to wandb if you want experiment tracking:"
echo "  wandb login"