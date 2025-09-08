#!/usr/bin/env python3
"""
Improved Training Configuration for Better Finnish-English Translation
This fixes the core training issues causing poor translation quality
"""

# Updated configuration for better learning
IMPROVED_CONFIG = {
    # Model Architecture - Even simpler for better convergence
    'd_model': 128,          # Further reduced from 256 to 128
    'num_heads': 4,          # Reduced from 8 to 4 for simpler attention
    'num_encoder_layers': 3, # Reduced from 4 to 3
    'num_decoder_layers': 3, # Reduced from 4 to 3
    'd_ff': 512,            # Reduced from 1024 to 512
    'max_seq_len': 64,      # Reduced from 128 to 64 for shorter sequences
    'dropout': 0.1,         # Reduced from 0.2 to 0.1 for less regularization

    # Training Hyperparameters - More conservative for stable learning
    'batch_size': 32,       # Increased from 16 to 32 for stable gradients
    'learning_rate': 5e-4,  # Reduced from 1e-3 to 5e-4 for stable learning
    'num_epochs': 25,       # Increased from 15 to 25 for more training time
    'warmup_steps': 1000,   # Increased from 500 to 1000 for gradual warmup

    # Loss and Regularization
    'label_smoothing': 0.05,   # Small amount of label smoothing for better generalization
    'use_label_smoothing': True,  # Enable label smoothing
    'weight_decay': 1e-5,      # Reduced weight decay
    'gradient_clip': 1.0,      # Increased gradient clipping for stability

    # Data Processing
    'vocab_size': 8000,     # Reduced from 10000 to 8000 for smaller vocabulary
    'min_freq': 3,          # Increased minimum frequency for vocabulary

    # Evaluation and Saving
    'save_every': 500,      # More frequent saving
    'eval_every': 100,      # More frequent evaluation

    # Other
    'pos_encoding_type': 'rope',
}

# Key improvements explained:
IMPROVEMENTS = {
    "Smaller Model": "Reduced complexity to ensure the model can actually learn the task",
    "Conservative Learning": "Lower learning rate with longer warmup for stable convergence",
    "More Training": "25 epochs instead of 15 to give model more time to learn",
    "Better Evaluation": "More frequent evaluation to catch overfitting early",
    "Smaller Vocabulary": "8K vocab instead of 10K to focus on common words",
    "Light Label Smoothing": "5% smoothing to improve generalization without hurting convergence"
}

def print_config_comparison():
    """Print comparison between old and new configs"""
    print("🔄 TRAINING CONFIGURATION IMPROVEMENTS")
    print("="*60)

    # Original config from kaggle_train.py
    original = {
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'd_ff': 1024,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'num_epochs': 15,
        'warmup_steps': 500,
        'vocab_size': 10000,
        'label_smoothing': 0.0,
        'use_label_smoothing': False
    }

    improved = IMPROVED_CONFIG

    print("📊 Key Changes:")
    changes = [
        ('d_model', original['d_model'], improved['d_model'], 'Simpler model'),
        ('num_heads', original['num_heads'], improved['num_heads'], 'Fewer attention heads'),
        ('layers', original['num_encoder_layers'], improved['num_encoder_layers'], 'Shallower network'),
        ('batch_size', original['batch_size'], improved['batch_size'], 'Larger batches'),
        ('learning_rate', original['learning_rate'], improved['learning_rate'], 'More conservative'),
        ('epochs', original['num_epochs'], improved['num_epochs'], 'More training time'),
        ('vocab_size', original['vocab_size'], improved['vocab_size'], 'Smaller vocabulary'),
        ('label_smoothing', original['label_smoothing'], improved['label_smoothing'], 'Light regularization')
    ]

    for param, old_val, new_val, reason in changes:
        print(f"  {param:15}: {old_val:>6} → {new_val:<6} ({reason})")

    print(f"\n💡 Why these changes help:")
    for improvement, explanation in IMPROVEMENTS.items():
        print(f"  • {improvement}: {explanation}")

    # Calculate parameter reduction
    old_params = original['d_model'] * original['d_model'] * (
        original['num_encoder_layers'] + original['num_decoder_layers']
    )
    new_params = improved['d_model'] * improved['d_model'] * (
        improved['num_encoder_layers'] + improved['num_decoder_layers']
    )

    reduction = (old_params - new_params) / old_params * 100
    print(f"\n📉 Parameter reduction: ~{reduction:.0f}% fewer parameters")
    print(f"   This makes the model easier to train and less prone to overfitting")

def create_improved_training_script():
    """Create an improved version of kaggle_train.py"""
    print("\n🔧 CREATING IMPROVED TRAINING SCRIPT")
    print("="*60)

    script_content = f'''#!/usr/bin/env python3
"""
IMPROVED Transformer Training Script for Kaggle
Fixed configuration for better Finnish-English translation learning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Kaggle paths
KAGGLE_INPUT_BASE = "/kaggle/input"
DATA_PATH = f"{{KAGGLE_INPUT_BASE}}/anlp-ass1"
CODE_PATH = f"{{KAGGLE_INPUT_BASE}}/anlp-ass1"

sys.path.append(DATA_PATH)
sys.path.append(CODE_PATH)
sys.path.append("/kaggle/working")

# Import modules
try:
    from encoder import TransformerEncoder
    from decoder import TransformerDecoder, Transformer
    from utils import (
        load_data, split_data, create_vocabulary, TransformerDataset,
        create_padding_mask, create_look_ahead_mask, calculate_bleu,
        indices_to_sentence, Vocabulary
    )
    print("✅ Successfully imported custom modules")
    modules_imported = True
except ImportError as e:
    print(f"❌ Error importing modules: {{e}}")
    modules_imported = False

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        assert pred.size(1) == self.classes
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.ignore_index >= 0:
            true_dist[:, self.ignore_index] = 0
            mask = (target == self.ignore_index).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)

        log_probs = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)

        if self.ignore_index >= 0:
            non_ignored = (target != self.ignore_index).float()
            loss = loss * non_ignored
            return loss.sum() / non_ignored.sum()
        else:
            return loss.mean()

def train_model():
    if not modules_imported:
        print("❌ Cannot proceed without required modules")
        return None

    # IMPROVED CONFIGURATION
    config = {IMPROVED_CONFIG}

    print("🚀 Starting IMPROVED Transformer Training")
    print("="*60)
    print("🎯 Key Improvements:")
    print("   • Simpler architecture for better convergence")
    print("   • Conservative learning rate with longer warmup")
    print("   • More training epochs for better learning")
    print("   • Smaller vocabulary focused on common words")
    print("   • Light label smoothing for generalization")
    print("="*60)
    print(f"Configuration: {{json.dumps(config, indent=2)}}")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {{device}}")

    # Load and prepare data
    print("\\n📁 Loading data...")
    try:
        en_file = f"{{DATA_PATH}}/EUbookshop.en"
        fi_file = f"{{DATA_PATH}}/EUbookshop.fi"

        if not os.path.exists(en_file) or not os.path.exists(fi_file):
            print(f"❌ Data files not found")
            return

        en_sentences, fi_sentences = load_data(en_file, fi_file)
        print(f"✅ Loaded {{len(en_sentences)}} sentence pairs")

        # Filter for shorter sentences to help learning
        max_len = 40  # Shorter than before
        filtered_en, filtered_fi = [], []
        for en, fi in zip(en_sentences, fi_sentences):
            if len(en.split()) <= max_len and len(fi.split()) <= max_len:
                filtered_en.append(en)
                filtered_fi.append(fi)

        print(f"✅ Filtered to {{len(filtered_en)}} shorter sentence pairs (max {{max_len}} words)")
        en_sentences, fi_sentences = filtered_en, filtered_fi

    except Exception as e:
        print(f"❌ Error loading data: {{e}}")
        return

    # Split data
    train_data, val_data, test_data = split_data(
        en_sentences, fi_sentences, train_ratio=0.8, val_ratio=0.1
    )
    print(f"   Train: {{len(train_data[0])}} pairs")
    print(f"   Val:   {{len(val_data[0])}} pairs")
    print(f"   Test:  {{len(test_data[0])}} pairs")

    # Create vocabularies with higher min_freq
    print("\\n📝 Creating vocabularies...")
    src_vocab = create_vocabulary(train_data[1], vocab_size=config['vocab_size'], min_freq=config['min_freq'])
    tgt_vocab = create_vocabulary(train_data[0], vocab_size=config['vocab_size'], min_freq=config['min_freq'])
    print(f"   Source vocab size: {{len(src_vocab)}}")
    print(f"   Target vocab size: {{len(tgt_vocab)}}")

    # Create datasets
    train_dataset = TransformerDataset(
        train_data[1], train_data[0], src_vocab, tgt_vocab, config['max_seq_len']
    )
    val_dataset = TransformerDataset(
        val_data[1], val_data[0], src_vocab, tgt_vocab, config['max_seq_len']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=val_dataset.collate_fn
    )

    # Initialize model
    print("\\n🤖 Initializing SIMPLER model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pos_encoding_type=config['pos_encoding_type']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {{total_params:,}} (much smaller for better learning)")

    # Loss and optimizer
    if config['use_label_smoothing']:
        criterion = LabelSmoothingLoss(
            classes=len(tgt_vocab),
            smoothing=config['label_smoothing'],
            ignore_index=tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN)
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0)
    )

    def lr_schedule(step):
        d_model = config['d_model']
        warmup_steps = config['warmup_steps']
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    train_losses = []
    val_losses = []
    step = 0
    best_val_loss = float('inf')

    print("\\n🏋️ Starting IMPROVED training...")
    print("="*60)

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {{epoch+1}}/{{config['num_epochs']}}")

        for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            optimizer.zero_grad()

            # Create masks
            src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))
            tgt_mask = create_look_ahead_mask(tgt_input, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
            optimizer.step()
            scheduler.step()

            step += 1
            epoch_train_loss += loss.item()
            train_losses.append(loss.item())

            progress_bar.set_postfix({{
                'Loss': f'{{loss.item():.4f}}',
                'LR': f'{{scheduler.get_last_lr()[0]:.2e}}',
                'Step': step
            }})

            # Validation
            if step % config['eval_every'] == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_src, val_tgt_input, val_tgt_output in val_loader:
                        val_src, val_tgt_input, val_tgt_output = val_src.to(device), val_tgt_input.to(device), val_tgt_output.to(device)
                        val_src_mask = create_padding_mask(val_src, src_vocab.get_idx(src_vocab.PAD_TOKEN))
                        val_tgt_mask = create_look_ahead_mask(val_tgt_input, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))
                        val_output = model(val_src, val_tgt_input, val_src_mask, val_tgt_mask)
                        val_loss += criterion(val_output.reshape(-1, val_output.size(-1)), val_tgt_output.reshape(-1)).item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                print(f"\\n📊 Step {{step}} - Val Loss: {{val_loss:.4f}}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({{
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'train_loss': epoch_train_loss / (batch_idx + 1),
                        'val_loss': val_loss,
                        'config': config,
                        'src_vocab': src_vocab,
                        'tgt_vocab': tgt_vocab,
                        'args': config
                    }}, '/kaggle/working/improved_best_model.pt')
                    print(f"💾 Saved IMPROVED model (val_loss: {{val_loss:.4f}})")

                model.train()

        # End of epoch
        epoch_train_loss /= len(train_loader)
        print(f"\\n📈 Epoch {{epoch+1}} Summary:")
        print(f"   Train Loss: {{epoch_train_loss:.4f}}")
        print(f"   Best Val Loss: {{best_val_loss:.4f}}")

        # Save checkpoint
        torch.save({{
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'train_loss': epoch_train_loss,
            'val_loss': best_val_loss,
            'config': config,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'args': config,
            'train_losses': train_losses,
            'val_losses': val_losses
        }}, f'/kaggle/working/improved_checkpoint_epoch_{{epoch+1}}.pt')

    # Final save
    torch.save({{
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': config['num_epochs'],
        'step': step,
        'train_loss': epoch_train_loss,
        'val_loss': best_val_loss,
        'config': config,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'args': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }}, '/kaggle/working/improved_final_model.pt')

    print("\\n🎉 IMPROVED Training completed!")
    print(f"💾 Models saved with 'improved_' prefix")
    print(f"📊 Best validation loss: {{best_val_loss:.4f}}")

    return model, src_vocab, tgt_vocab, config

if __name__ == "__main__":
    train_model()
'''

    with open('kaggle_train_improved.py', 'w') as f:
        f.write(script_content)

    print("✅ Created 'kaggle_train_improved.py'")
    print("📋 To use this:")
    print("   1. Upload this script to Kaggle")
    print("   2. Run it instead of the original kaggle_train.py")
    print("   3. It will create models with 'improved_' prefix")
    print("   4. Use 'improved_best_model.pt' for testing")

if __name__ == "__main__":
    print_config_comparison()
    create_improved_training_script()
