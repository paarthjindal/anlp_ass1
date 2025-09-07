#!/usr/bin/env python3
"""
Colab Training Script for Transformer Machine Translation
Modified to work with Google Drive mounted data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append('/content/drive/MyDrive/')  # Adjust this path based on your Google Drive structure

# Import your modules (adjust paths as needed)
from encoder import TransformerEncoder
from decoder import TransformerDecoder, Transformer
from utils import (
    load_data, split_data, create_vocabulary, TransformerDataset,
    create_padding_mask, create_look_ahead_mask, calculate_bleu,
    indices_to_sentence, LabelSmoothing
)

def parse_args():
    """Parse command line arguments for Colab training"""
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation in Colab')

    # Data arguments - Updated for Google Drive paths
    parser.add_argument('--drive_path', type=str, default='/content/drive/MyDrive',
                        help='Path to mounted Google Drive')
    parser.add_argument('--project_folder', type=str, default='',
                        help='Project folder name in Google Drive')
    parser.add_argument('--data_dir', type=str, default='EUbookshop',
                        help='Directory containing the dataset')
    parser.add_argument('--src_file', type=str, default='EUbookshop.fi',
                        help='Source language file (Finnish)')
    parser.add_argument('--tgt_file', type=str, default='EUbookshop.en',
                        help='Target language file (English)')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension (reduced for Colab GPU)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4,
                        help='Number of encoder layers (reduced for memory)')
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                        help='Number of decoder layers (reduced for memory)')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension (reduced for memory)')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length (reduced for memory)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pos_encoding_type', type=str, default='rope',
                        choices=['rope', 'relative_bias', 'sinusoidal'],
                        help='Type of positional encoding')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduced for Colab GPU)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps for learning rate scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Save arguments
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model checkpoints in Google Drive')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save model every N epochs')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log training progress every N steps')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    return parser.parse_args()

class ColabTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Setup paths for Google Drive
        self.project_path = Path(args.drive_path) / args.project_folder
        self.data_path = self.project_path / args.data_dir
        self.save_path = self.project_path / args.save_dir

        # Create save directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"Project path: {self.project_path}")
        print(f"Data path: {self.data_path}")
        print(f"Save path: {self.save_path}")
        print(f"Using device: {self.device}")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")

        self.setup_data()
        self.setup_model()
        self.setup_training()

    def setup_data(self):
        """Load and prepare data from Google Drive"""
        print("Loading data from Google Drive...")

        src_file_path = self.data_path / self.args.src_file
        tgt_file_path = self.data_path / self.args.tgt_file

        # Check if files exist
        if not src_file_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_file_path}")
        if not tgt_file_path.exists():
            raise FileNotFoundError(f"Target file not found: {tgt_file_path}")

        # Load data
        en_sentences, fi_sentences = load_data(str(tgt_file_path), str(src_file_path))
        print(f"Loaded {len(fi_sentences)} sentence pairs")

        # Split data
        self.train_data, self.val_data, self.test_data = split_data(en_sentences, fi_sentences, train_ratio=0.8, val_ratio=0.1)
        print(f"Train: {len(self.train_data[0])}, Val: {len(self.val_data[0])}, Test: {len(self.test_data[0])}")

        # Create vocabularies
        print("Creating vocabularies...")
        self.src_vocab = create_vocabulary(self.train_data[1])  # Finnish (source)
        self.tgt_vocab = create_vocabulary(self.train_data[0])  # English (target)

        print(f"Source vocabulary size: {len(self.src_vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_vocab)}")

        # Create datasets
        self.train_dataset = TransformerDataset(
            self.train_data[1], self.train_data[0], self.src_vocab, self.tgt_vocab, self.args.max_seq_len
        )
        self.val_dataset = TransformerDataset(
            self.val_data[1], self.val_data[0], self.src_vocab, self.tgt_vocab, self.args.max_seq_len
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2
        )

    def setup_model(self):
        """Initialize the Transformer model"""
        print(f"Creating model with {self.args.pos_encoding_type} positional encoding...")

        self.model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=self.args.d_model,
            num_heads=self.args.num_heads,
            num_encoder_layers=self.args.num_encoder_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            d_ff=self.args.d_ff,
            max_seq_len=self.args.max_seq_len,
            dropout=self.args.dropout,
            pos_encoding_type=self.args.pos_encoding_type
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) ** -0.5,
                (step + 1) * self.args.warmup_steps ** -1.5
            )
        )

        # Loss function with label smoothing
        self.criterion = LabelSmoothing(
            size=len(self.tgt_vocab),
            padding_idx=self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN),
            smoothing=self.args.label_smoothing
        )

        print("Training setup completed!")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for step, batch in enumerate(progress_bar):
            src, tgt_input, tgt_output = [b.to(self.device) for b in batch]

            # Create masks
            src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))
            tgt_mask = create_look_ahead_mask(tgt_input, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # Calculate loss
            loss = self.criterion(output, tgt_output)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            if step % self.args.log_every == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                src, tgt_input, tgt_output = [b.to(self.device) for b in batch]

                # Create masks
                src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))
                tgt_mask = create_look_ahead_mask(tgt_input, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                # Forward pass
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output, tgt_output)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint to Google Drive"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(self.args),
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab
        }

        checkpoint_path = self.save_path / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.save_path / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved: {best_model_path}")

    def train(self):
        """Main training loop"""
        print(f"Starting training with {self.args.pos_encoding_type} positional encoding...")
        print(f"Total epochs: {self.args.num_epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Model dimension: {self.args.d_model}")

        for epoch in range(1, self.args.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if epoch % self.args.save_every == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Training completed!")

def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check if running in Colab
    try:
        import google.colab
        print("Running in Google Colab")
        # Mount Google Drive if not already mounted
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
    except ImportError:
        print("Not running in Google Colab")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create trainer and start training
    trainer = ColabTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
