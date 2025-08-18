import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (load_data, split_data, create_vocabulary, TransformerDataset,
                   save_model, LabelSmoothing, calculate_bleu, indices_to_sentence)
from decoder import Transformer
from encoder import TransformerEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./EUbookshop/',
                        help='Directory containing the dataset')
    parser.add_argument('--src_file', type=str, default='EUbookshop.fi',
                        help='Source language file (Finnish)')
    parser.add_argument('--tgt_file', type=str, default='EUbookshop.en',
                        help='Target language file (English)')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pos_encoding_type', type=str, default='rope',
                        choices=['rope', 'relative_bias', 'sinusoidal'],
                        help='Type of positional encoding')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps for learning rate scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log training progress every N steps')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    return parser.parse_args()

class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler,
                 src_vocab, tgt_vocab, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.args.device)
            tgt = tgt.to(self.args.device)

            # Prepare decoder input and target
            tgt_input = tgt[:, :-1]  # All tokens except last
            tgt_output = tgt[:, 1:]  # All tokens except first

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)

            # Calculate loss
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = self.criterion(output, tgt_output)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            # Update parameters
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log progress
            if batch_idx % self.args.log_every == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc='Validation', leave=False):
                src = src.to(self.args.device)
                tgt = tgt.to(self.args.device)

                # Prepare decoder input and target
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Forward pass
                output = self.model(src, tgt_input)

                # Calculate loss
                output = output.contiguous().view(-1, output.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)

                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()

        return total_loss / num_batches

    def train(self):
        print(f"Starting training with {self.args.pos_encoding_type} positional encoding...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        for epoch in range(self.args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch, is_best=True)

            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_model(epoch)

            # Save training curves
            self.plot_training_curves()

    def save_model(self, epoch, is_best=False):
        os.makedirs(self.args.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'args': self.args,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab
        }

        if is_best:
            filename = f"best_model_{self.args.pos_encoding_type}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}_{self.args.pos_encoding_type}.pt"

        filepath = os.path.join(self.args.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved model to {filepath}")

    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.args.pos_encoding_type.upper()}')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, f'training_curves_{self.args.pos_encoding_type}.png'))
        plt.close()

class NoamLR:
    """Learning rate scheduler from the original Transformer paper"""
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5),
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and prepare data
    print("Loading data...")
    src_file = os.path.join(args.data_dir, args.src_file)
    tgt_file = os.path.join(args.data_dir, args.tgt_file)

    src_sentences, tgt_sentences = load_data(src_file, tgt_file)
    print(f"Loaded {len(src_sentences)} sentence pairs")

    # Split data
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = split_data(
        src_sentences, tgt_sentences, train_ratio=0.8, val_ratio=0.1
    )

    print(f"Train: {len(train_src)}, Val: {len(val_src)}, Test: {len(test_src)}")

    # Create vocabularies
    print("Creating vocabularies...")
    src_vocab = create_vocabulary(train_src, min_freq=2)
    tgt_vocab = create_vocabulary(train_tgt, min_freq=2)

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    # Create datasets
    train_dataset = TransformerDataset(train_src, train_tgt, src_vocab, tgt_vocab, args.max_seq_len)
    val_dataset = TransformerDataset(val_src, val_tgt, src_vocab, tgt_vocab, args.max_seq_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"Creating model with {args.pos_encoding_type} positional encoding...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding_type
    )

    model.to(args.device)

    # Create criterion with label smoothing
    criterion = LabelSmoothing(
        size=len(tgt_vocab),
        padding_idx=tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN),
        smoothing=args.label_smoothing
    )

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                          betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, args.d_model, args.warmup_steps)

    # Create trainer and train
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        args=args
    )

    trainer.train()

    print("Training completed!")

    # Save final training information
    training_info = {
        'pos_encoding_type': args.pos_encoding_type,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'args': vars(args)
    }

    with open(os.path.join(args.save_dir, f'training_info_{args.pos_encoding_type}.json'), 'w') as f:
        json.dump(training_info, f, indent=2)

if __name__ == '__main__':
    main()
