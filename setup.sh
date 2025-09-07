#!/bin/bash

echo "Setting up Transformer Machine Translation Environment"
echo "====================================================="

# Install required packages
echo "Installing required Python packages..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install matplotlib seaborn tqdm sacrebleu numpy

echo "Installation completed!"

# Test imports
echo "Testing imports..."
python3 -c "
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
print('All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
"

echo "Setup completed successfully!"
