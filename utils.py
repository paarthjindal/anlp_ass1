import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sacrebleu import corpus_bleu

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        # Add special tokens
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def get_idx(self, word):
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])

    def get_word(self, idx):
        return self.idx2word.get(idx, self.UNK_TOKEN)

def create_vocabulary(sentences: List[str], vocab_size: int = 10000, min_freq: int = 2) -> Vocabulary:
    """Create vocabulary from sentences with proper size limiting"""
    vocab = Vocabulary()

    # Count words
    for sentence in sentences:
        vocab.add_sentence(sentence.lower().strip())

    # Create filtered vocabulary with size limit
    filtered_vocab = Vocabulary()

    # Always add special tokens first
    special_tokens = [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN, vocab.UNK_TOKEN]
    for token in special_tokens:
        filtered_vocab.add_word(token)

    # Sort words by frequency (most frequent first)
    word_freq_pairs = [(word, count) for word, count in vocab.word_count.items()
                       if word not in special_tokens and count >= min_freq]
    word_freq_pairs.sort(key=lambda x: x[1], reverse=True)

    # Add words up to vocab_size limit
    words_added = len(special_tokens)  # Start with special tokens count
    for word, count in word_freq_pairs:
        if words_added >= vocab_size:
            break
        filtered_vocab.add_word(word)
        words_added += 1

    print(f"   Created vocabulary: {len(filtered_vocab)} words (target: {vocab_size}, min_freq: {min_freq})")
    return filtered_vocab

def load_data(en_file: str, fi_file: str) -> Tuple[List[str], List[str]]:
    """Load parallel sentences from files"""
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f.readlines()]

    with open(fi_file, 'r', encoding='utf-8') as f:
        fi_sentences = [line.strip() for line in f.readlines()]

    # Filter empty sentences and ensure same length
    filtered_pairs = []
    for en, fi in zip(en_sentences, fi_sentences):
        if en.strip() and fi.strip():
            filtered_pairs.append((en.strip(), fi.strip()))

    en_sentences, fi_sentences = zip(*filtered_pairs)
    return list(en_sentences), list(fi_sentences)

def split_data(en_sentences: List[str], fi_sentences: List[str],
               train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train, validation, and test sets"""
    total_size = len(en_sentences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Shuffle data
    indices = np.random.permutation(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_en = [en_sentences[i] for i in train_indices]
    train_fi = [fi_sentences[i] for i in train_indices]

    val_en = [en_sentences[i] for i in val_indices]
    val_fi = [fi_sentences[i] for i in val_indices]

    test_en = [en_sentences[i] for i in test_indices]
    test_fi = [fi_sentences[i] for i in test_indices]

    return (train_en, train_fi), (val_en, val_fi), (test_en, test_fi)

def sentence_to_indices(sentence: str, vocab: Vocabulary, max_length: int = None):
    """Convert sentence to indices"""
    words = sentence.lower().strip().split()
    indices = [vocab.get_idx(word) for word in words]

    if max_length:
        if len(indices) > max_length - 2:  # -2 for SOS and EOS
            indices = indices[:max_length - 2]

        # Add SOS and EOS
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad if necessary
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

    return indices

def indices_to_sentence(indices: List[int], vocab: Vocabulary):
    """Convert indices back to sentence"""
    words = []
    for idx in indices:
        word = vocab.get_word(idx)
        if word == vocab.EOS_TOKEN:
            break
        if word not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN]:
            words.append(word)
    return ' '.join(words)

def create_padding_mask(seq, pad_idx):
    """Create padding mask for sequences"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(seq, pad_idx):
    """Create look-ahead mask for decoder"""
    seq_len = seq.size(1)

    # Create padding mask
    pad_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    # Create look-ahead mask
    look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len, device=seq.device)).bool()
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    # Combine both masks
    mask = pad_mask & look_ahead_mask

    return mask

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        src_indices = sentence_to_indices(src_sentence, self.src_vocab, self.max_length)
        tgt_indices = sentence_to_indices(tgt_sentence, self.tgt_vocab, self.max_length)

        # For transformer training, we need:
        # - tgt_input: target sequence for decoder input (includes <SOS>)
        # - tgt_output: target sequence for loss calculation (includes <EOS>)
        tgt_input = tgt_indices[:-1]  # Remove last token (usually <EOS>)
        tgt_output = tgt_indices[1:]  # Remove first token (usually <SOS>)

        return torch.tensor(src_indices), torch.tensor(tgt_input), torch.tensor(tgt_output)

    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)

        # Stack tensors
        src_batch = torch.stack(src_batch)
        tgt_input_batch = torch.stack(tgt_input_batch)
        tgt_output_batch = torch.stack(tgt_output_batch)

        return src_batch, tgt_input_batch, tgt_output_batch

def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score"""
    return corpus_bleu(predictions, [references]).score

def calculate_bert_score(predictions: List[str], references: List[str], model_type="distilbert-base-uncased", device=None):
    """
    Calculate BERTScore for predictions vs references

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        model_type: BERT model to use for scoring (default: distilbert for speed)
        device: Device to run on (auto-detected if None)

    Returns:
        dict: Contains precision, recall, f1 scores (both individual and average)
    """
    try:
        from bert_score import score

        # Calculate BERTScore
        P, R, F1 = score(predictions, references, model_type=model_type, device=device, verbose=False)

        # Convert to float and calculate averages
        precision_scores = P.tolist()
        recall_scores = R.tolist()
        f1_scores = F1.tolist()

        return {
            'precision_avg': float(P.mean()),
            'recall_avg': float(R.mean()),
            'f1_avg': float(F1.mean()),
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores
        }

    except ImportError:
        print("⚠️  BERTScore not available. Install with: pip install bert-score")
        return {
            'precision_avg': 0.0,
            'recall_avg': 0.0,
            'f1_avg': 0.0,
            'precision_scores': [0.0] * len(predictions),
            'recall_scores': [0.0] * len(predictions),
            'f1_scores': [0.0] * len(predictions)
        }
    except Exception as e:
        print(f"❌ Error calculating BERTScore: {e}")
        return {
            'precision_avg': 0.0,
            'recall_avg': 0.0,
            'f1_avg': 0.0,
            'precision_scores': [0.0] * len(predictions),
            'recall_scores': [0.0] * len(predictions),
            'f1_scores': [0.0] * len(predictions)
        }

def plot_training_curves(train_losses, val_losses, title="Training Curves"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_model(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

class LabelSmoothing(nn.Module):
    """Label smoothing for better training"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # x should be logits [batch_size, seq_len, vocab_size]
        # target should be indices [batch_size, seq_len]

        # Reshape for processing
        batch_size, seq_len, vocab_size = x.size()
        x = x.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        target = target.view(-1)    # [batch_size * seq_len]

        # Apply log softmax to get log probabilities
        x = F.log_softmax(x, dim=1)

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
