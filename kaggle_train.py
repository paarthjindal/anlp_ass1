#!/usr/bin/env python3
"""
Transformer Training Script for Kaggle
This script trains a Finnish-English transformer model
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

# Kaggle paths - adjust these to match your input dataset names
KAGGLE_INPUT_BASE = "/kaggle/input"
# Replace 'your-dataset-name' with your actual dataset name in Kaggle
DATA_PATH = f"{KAGGLE_INPUT_BASE}/anlp-ass1"  # Update this!
CODE_PATH = f"{KAGGLE_INPUT_BASE}/anlp-ass1"  # If you uploaded code separately

# Add paths for imports
sys.path.append(DATA_PATH)
sys.path.append(CODE_PATH)
sys.path.append("/kaggle/working")

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization in translation tasks.

    Args:
        classes: number of classes (vocabulary size)
        smoothing: smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        ignore_index: index to ignore (padding token)
    """
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: predictions (batch_size * seq_len, vocab_size)
            target: true labels (batch_size * seq_len)
        """
        assert pred.size(1) == self.classes

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Ignore padding tokens
        if self.ignore_index >= 0:
            true_dist[:, self.ignore_index] = 0
            mask = (target == self.ignore_index).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)

        # Calculate KL divergence
        log_probs = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)

        # Only average over non-ignored tokens
        if self.ignore_index >= 0:
            non_ignored = (target != self.ignore_index).float()
            loss = loss * non_ignored
            return loss.sum() / non_ignored.sum()
        else:
            return loss.mean()

# Import your modules (assuming they're in your Kaggle input)
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
    print(f"❌ Error importing modules: {e}")
    print("Please ensure your encoder.py, decoder.py, and utils.py are in the Kaggle input")
    print("Available files in DATA_PATH:")
    try:
        files = os.listdir(DATA_PATH)
        for f in sorted(files):
            print(f"  {f}")
    except:
        print("  Could not list files")
    modules_imported = False

def train_model():
    """Main training function"""

    # Check if modules were imported successfully
    if not modules_imported:
        print("❌ Cannot proceed without required modules")
        print("📋 To fix this:")
        print("1. Make sure encoder.py, decoder.py, and utils.py are uploaded to your Kaggle dataset")
        print("2. Check that the files are in the correct path")
        print("3. Ensure sacrebleu is installed (script will try to install it)")
        return None

    # Configuration
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 128,
        'dropout': 0.1,
        'pos_encoding_type': 'rope',  # or 'sinusoidal'
        'batch_size': 32,
        'learning_rate': 2e-4,  # Increased from 1e-4 to 2e-4
        'num_epochs': 10,  # Increased from 5 to 10
        'warmup_steps': 2000,  # Reduced from 4000 to 2000
        'vocab_size': 10000,
        'save_every': 1000,
        'eval_every': 500,
        'label_smoothing': 0.05,  # Reduced from 0.1 to 0.05 (5% smoothing)
        'use_label_smoothing': False  # Try Cross Entropy instead of Label Smoothing
    }

    print("🚀 Starting Transformer Training on Kaggle")
    print("="*60)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print("="*60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU not available - using CPU")
        print("   💡 In Kaggle, make sure to enable GPU in Settings > Accelerator")
        print("   💡 Then restart the session and run again")

    # Load data
    print("\n📁 Loading data...")
    try:
        # Update these paths to match your Kaggle dataset structure
        en_file = f"{DATA_PATH}/EUbookshop.en"
        fi_file = f"{DATA_PATH}/EUbookshop.fi"

        if not os.path.exists(en_file) or not os.path.exists(fi_file):
            print(f"❌ Data files not found:")
            print(f"   Looking for: {en_file}")
            print(f"   Looking for: {fi_file}")
            print("   Please upload your data to Kaggle input section")
            return

        en_sentences, fi_sentences = load_data(en_file, fi_file)
        print(f"✅ Loaded {len(en_sentences)} sentence pairs")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Split data
    print("\n🔀 Splitting data...")
    train_data, val_data, test_data = split_data(
        en_sentences, fi_sentences,
        train_ratio=0.8, val_ratio=0.1
    )
    print(f"   Train: {len(train_data[0])} pairs")
    print(f"   Val:   {len(val_data[0])} pairs")
    print(f"   Test:  {len(test_data[0])} pairs")

    # Create vocabularies
    print("\n📝 Creating vocabularies...")
    src_vocab = create_vocabulary(train_data[1], vocab_size=config['vocab_size'], min_freq=2)  # Finnish
    tgt_vocab = create_vocabulary(train_data[0], vocab_size=config['vocab_size'], min_freq=2)  # English

    print(f"   Source vocab size: {len(src_vocab)}")
    print(f"   Target vocab size: {len(tgt_vocab)}")

    # Create datasets
    print("\n🗂️ Creating datasets...")
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

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    # Initialize model
    print("\n🤖 Initializing model...")
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    if config['use_label_smoothing']:
        print(f"🎯 Using Label Smoothing Loss (smoothing: {config['label_smoothing']})")
        criterion = LabelSmoothingLoss(
            classes=len(tgt_vocab),
            smoothing=config['label_smoothing'],
            ignore_index=tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN)
        )
    else:
        print("🎯 Using Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler
    def lr_schedule(step):
        """Learning rate schedule with warmup - handles step=0 case"""
        d_model = config['d_model']
        warmup_steps = config['warmup_steps']

        # Handle step=0 case to avoid division by zero
        step = max(1, step)

        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training tracking
    train_losses = []
    val_losses = []
    step = 0
    best_val_loss = float('inf')

    print("\n🏋️ Starting training...")
    print("="*60)

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Create masks
            src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))
            tgt_mask = create_look_ahead_mask(tgt_input, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

            # Model forward
            output = model(src, tgt_input, src_mask, tgt_mask)

            # Calculate loss
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            step += 1
            epoch_train_loss += loss.item()
            train_losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                'Step': step
            })

            # Validation evaluation
            if step % config['eval_every'] == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_src, val_tgt_input, val_tgt_output in val_loader:
                        val_src, val_tgt_input, val_tgt_output = val_src.to(device), val_tgt_input.to(device), val_tgt_output.to(device)

                        val_src_mask = create_padding_mask(val_src, src_vocab.get_idx(src_vocab.PAD_TOKEN))
                        val_tgt_mask = create_look_ahead_mask(val_tgt_input, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

                        val_output = model(val_src, val_tgt_input, val_src_mask, val_tgt_mask)
                        val_loss += criterion(
                            val_output.reshape(-1, val_output.size(-1)),
                            val_tgt_output.reshape(-1)
                        ).item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                print(f"\n📊 Step {step} - Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
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
                        'args': config  # For compatibility
                    }, '/kaggle/working/best_model.pt')
                    print(f"💾 Saved new best model (val_loss: {val_loss:.4f})")

                model.train()

        # End of epoch summary
        epoch_train_loss /= len(train_loader)
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {epoch_train_loss:.4f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint every epoch
        torch.save({
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
        }, f'/kaggle/working/checkpoint_epoch_{epoch+1}.pt')

    # Final model save
    torch.save({
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
    }, '/kaggle/working/final_model.pt')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    eval_steps = list(range(config['eval_every'], step + 1, config['eval_every']))
    plt.plot(eval_steps[:len(val_losses)], val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n🎉 Training completed!")
    print(f"💾 Models saved to /kaggle/working/")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🔧 Total parameters: {total_params:,}")

    return model, src_vocab, tgt_vocab, config

if __name__ == "__main__":
    # Update these paths to match your Kaggle dataset
    print("🔧 IMPORTANT: Update the DATA_PATH variable to match your Kaggle dataset name!")
    print(f"Current DATA_PATH: {DATA_PATH}")
    print("Change 'your-dataset-name' to your actual dataset name in Kaggle")
    print("="*60)

    try:
        model, src_vocab, tgt_vocab, config = train_model()
        if model is not None:
            print("✅ Training script completed successfully!")
        else:
            print("❌ Training failed - check error messages above")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
