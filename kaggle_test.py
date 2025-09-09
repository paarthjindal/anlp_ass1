#!/usr/bin/env python3
"""
Transformer Testing Script for Kaggle
This script loads a trained model and evaluates it on test data
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Kaggle paths - matches your input structure
KAGGLE_INPUT_BASE = "/kaggle/input"
DATA_PATH = f"{KAGGLE_INPUT_BASE}/anlp-ass1"  # Your dataset name

# Add paths for imports
sys.path.append(DATA_PATH)
sys.path.append("/kaggle/working")

print(f"📁 Data path: {DATA_PATH}")
print(f"📁 Working directory: /kaggle/working")

# Import your modules
try:
    from encoder import TransformerEncoder
    from decoder import TransformerDecoder, Transformer
    from utils import (
        load_data, split_data, create_vocabulary, TransformerDataset,
        create_padding_mask, create_look_ahead_mask, calculate_bleu, calculate_bert_score,
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

# Add safe globals for PyTorch loading
try:
    torch.serialization.add_safe_globals([Vocabulary])
except Exception:
    pass

class ModelTester:
    def __init__(self, model_path, device='cuda'):
        """Initialize the model tester for Kaggle"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # Load checkpoint
        print(f"📥 Loading model from {model_path}")

        # Handle PyTorch's weights_only security feature
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading with weights_only=False: {e}")
            try:
                torch.serialization.add_safe_globals([Vocabulary])
                self.checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e2:
                print(f"Error loading with safe globals: {e2}")
                raise e2

        # Extract vocabulary and configuration
        self.src_vocab = self.checkpoint['src_vocab']
        self.tgt_vocab = self.checkpoint['tgt_vocab']
        self.config = self.checkpoint['args']

        # Initialize model with relative attention parameters if available
        model_kwargs = {
            'src_vocab_size': len(self.src_vocab),
            'tgt_vocab_size': len(self.tgt_vocab),
            'd_model': self.config['d_model'],
            'num_heads': self.config['num_heads'],
            'num_encoder_layers': self.config['num_encoder_layers'],
            'num_decoder_layers': self.config['num_decoder_layers'],
            'd_ff': self.config['d_ff'],
            'max_seq_len': self.config['max_seq_len'],
            'dropout': self.config['dropout'],
            'pos_encoding_type': self.config['pos_encoding_type']
        }

        # Add relative attention parameters if they exist in config (for relative_bias models)
        if 'relative_attention_num_buckets' in self.config:
            model_kwargs['relative_attention_num_buckets'] = self.config['relative_attention_num_buckets']
        if 'relative_attention_max_distance' in self.config:
            model_kwargs['relative_attention_max_distance'] = self.config['relative_attention_max_distance']

        self.model = Transformer(**model_kwargs).to(self.device)

        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"🔧 Positional encoding: {self.config['pos_encoding_type']}")
        if self.config['pos_encoding_type'] == 'relative_bias':
            if 'relative_attention_num_buckets' in self.config:
                print(f"🔧 Relative attention buckets: {self.config['relative_attention_num_buckets']}")
            if 'relative_attention_max_distance' in self.config:
                print(f"🔧 Relative attention max distance: {self.config['relative_attention_max_distance']}")
        elif self.config['pos_encoding_type'] == 'rope':
            print(f"🔧 Using RoPE (Rotary Position Embeddings)")
        print(f"📊 Source vocab size: {len(self.src_vocab):,}")
        print(f"📊 Target vocab size: {len(self.tgt_vocab):,}")
        print(f"🤖 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def sentence_to_indices(self, sentence, vocab, max_length=None):
        """Convert sentence to indices"""
        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        if max_length is None:
            max_length = self.config['max_seq_len']

        if len(indices) > max_length - 2:
            indices = indices[:max_length - 2]

        # Add SOS and EOS
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad if necessary
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    def translate_greedy(self, sentence, max_length=50):
        """Translate using greedy decoding with UNK prevention"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        # Vocabulary indices
        unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
        sos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)
        eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
        pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)

            tgt_indices = [sos_idx]
            generated_tokens = set()  # Track generated tokens to prevent excessive repetition

            for step in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, pad_idx)

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = decoder_output[0, -1, :]

                # Apply UNK penalty (strong penalty to avoid UNK tokens)
                logits[unk_idx] -= 10.0

                # Apply repetition penalty for recently generated tokens
                for token_idx in list(generated_tokens)[-5:]:  # Penalize last 5 unique tokens
                    if token_idx in generated_tokens:
                        logits[token_idx] -= 1.0

                # Greedy selection (pick highest probability)
                next_token = torch.argmax(logits).item()

                tgt_indices.append(next_token)
                generated_tokens.add(next_token)

                if next_token == eos_idx:
                    break

        return indices_to_sentence(tgt_indices, self.tgt_vocab)

    def translate_top_k(self, sentence, max_length=50, k=50, p=0.9, temperature=1.0):
        """Translate using top-k sampling with nucleus sampling and UNK prevention"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        # Vocabulary indices
        unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
        sos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)
        eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
        pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)

            tgt_indices = [sos_idx]
            generated_tokens = set()  # Track generated tokens to prevent excessive repetition

            for step in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, pad_idx)

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = decoder_output[0, -1, :] / temperature

                # Apply UNK penalty (strong penalty to avoid UNK tokens)
                logits[unk_idx] -= 10.0

                # Apply repetition penalty for recently generated tokens
                for token_idx in list(generated_tokens)[-5:]:  # Penalize last 5 unique tokens
                    if token_idx in generated_tokens:
                        logits[token_idx] -= 1.0

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Apply top-k filtering
                if k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(k, probs.size(-1)))
                    # Zero out probabilities outside top-k
                    filtered_probs = torch.zeros_like(probs)
                    filtered_probs.scatter_(-1, top_k_indices, top_k_probs)
                    probs = filtered_probs

                # Apply nucleus (top-p) filtering
                if p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Find cutoff index for nucleus sampling
                    cutoff_index = torch.where(cumulative_probs > p)[0]
                    if len(cutoff_index) > 0:
                        cutoff_index = cutoff_index[0].item()
                        # Zero out probabilities beyond the cutoff
                        sorted_probs[cutoff_index:] = 0
                        # Scatter back to original positions
                        probs = torch.zeros_like(probs)
                        probs.scatter_(-1, sorted_indices, sorted_probs)

                # Renormalize probabilities
                probs = probs / (probs.sum() + 1e-8)

                # Sample from the distribution
                if k == 1:  # Greedy decoding
                    next_token = torch.argmax(probs).item()
                else:  # Stochastic sampling
                    next_token = torch.multinomial(probs, 1).item()

                tgt_indices.append(next_token)
                generated_tokens.add(next_token)

                if next_token == eos_idx:
                    break

        return indices_to_sentence(tgt_indices, self.tgt_vocab)

    def translate_beam_search(self, sentence, beam_size=4, max_length=50):
        """Translate using beam search with UNK prevention and improved beam exploration"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        # Vocabulary indices
        unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
        sos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)
        eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
        pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)

            # Initialize beams with SOS token
            beams = [[sos_idx]]
            beam_scores = [0.0]
            completed_beams = []  # Store completed sequences

            for step in range(max_length):
                candidates = []

                # Process each active beam
                for i, beam in enumerate(beams):
                    # Skip if this beam is already completed
                    if beam[-1] == eos_idx:
                        completed_beams.append((beam_scores[i], beam))
                        continue

                    tgt = torch.tensor([beam], device=self.device)
                    tgt_mask = create_look_ahead_mask(tgt, pad_idx)

                    decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                    logits = decoder_output[0, -1, :]

                    # Apply UNK penalty (moderate penalty to avoid UNK tokens)
                    logits[unk_idx] -= 5.0  # Reduced from 10.0 to allow more exploration

                    # Apply mild repetition penalty for tokens already in the beam (except special tokens)
                    beam_tokens = set(beam)
                    for token_idx in beam_tokens:
                        if token_idx not in [sos_idx, eos_idx, pad_idx, unk_idx]:
                            logits[token_idx] -= 0.3  # Reduced penalty

                    # Encourage longer sequences by penalizing EOS early
                    if step < 3:  # Don't allow EOS in first 3 steps
                        logits[eos_idx] -= 10.0

                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get top beam_size candidates for this beam
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size * 2)  # Consider more options

                    for j in range(len(top_k_probs)):
                        new_token = top_k_indices[j].item()
                        new_beam = beam + [new_token]
                        new_score = beam_scores[i] + top_k_probs[j].item()

                        # Length normalization to prevent bias towards shorter sequences
                        normalized_score = new_score / len(new_beam)

                        candidates.append((normalized_score, new_score, new_beam))

                # Sort candidates by normalized score and select top beam_size
                candidates.sort(key=lambda x: x[0], reverse=True)

                # Select beams for next iteration
                beams = []
                beam_scores = []

                for normalized_score, raw_score, beam in candidates[:beam_size]:
                    if beam[-1] == eos_idx:
                        completed_beams.append((raw_score, beam))
                    else:
                        beams.append(beam)
                        beam_scores.append(raw_score)

                # If no active beams left, break
                if not beams:
                    break

                # If we have enough completed beams and they're good quality, we can stop
                if len(completed_beams) >= beam_size and step >= 5:
                    break

            # Add any remaining active beams to completed beams
            for i, beam in enumerate(beams):
                completed_beams.append((beam_scores[i], beam))

            # Select best completed beam
            if completed_beams:
                completed_beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # Length normalized
                best_beam = completed_beams[0][1]
            else:
                # Fallback to first beam if no completions
                best_beam = beams[0] if beams else [sos_idx, eos_idx]

        return indices_to_sentence(best_beam, self.tgt_vocab)

    def translate_beam_search_alternative(self, sentence, beam_size=8, max_length=50):
        """Alternative beam search with different parameters for comparison"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        # Vocabulary indices
        unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
        sos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)
        eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
        pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)

            # Initialize with SOS
            sequences = [[sos_idx]]
            scores = [0.0]

            for step in range(max_length):
                all_candidates = []

                for i, seq in enumerate(sequences):
                    if seq[-1] == eos_idx:
                        all_candidates.append((scores[i], seq))
                        continue

                    tgt = torch.tensor([seq], device=self.device)
                    tgt_mask = create_look_ahead_mask(tgt, pad_idx)

                    decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                    logits = decoder_output[0, -1, :]

                    # Light UNK penalty
                    logits[unk_idx] -= 3.0

                    # Prevent immediate EOS
                    if step < 2:
                        logits[eos_idx] -= 5.0

                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get all possible next tokens (not just top-k)
                    for token_idx in range(len(self.tgt_vocab)):
                        new_seq = seq + [token_idx]
                        new_score = scores[i] + log_probs[token_idx].item()
                        all_candidates.append((new_score, new_seq))

                # Select top beam_size candidates
                all_candidates.sort(key=lambda x: x[0], reverse=True)

                sequences = []
                scores = []

                for score, seq in all_candidates[:beam_size]:
                    sequences.append(seq)
                    scores.append(score)

                # Check if all sequences are complete
                if all(seq[-1] == eos_idx for seq in sequences):
                    break

            # Return best sequence
            best_idx = max(range(len(scores)), key=lambda i: scores[i] / len(sequences[i]))
            return indices_to_sentence(sequences[best_idx], self.tgt_vocab)

    def translate_conservative(self, sentence, max_length=50):
        """Conservative top-k strategy: k=20, p=0.8, temp=1.0"""
        return self.translate_top_k(sentence, max_length=max_length, k=20, p=0.8, temperature=1.0)

    def translate_balanced(self, sentence, max_length=50):
        """Balanced top-k strategy: k=50, p=0.9, temp=1.0"""
        return self.translate_top_k(sentence, max_length=max_length, k=50, p=0.9, temperature=1.0)

    def translate_creative(self, sentence, max_length=50):
        """Creative top-k strategy: k=100, p=0.95, temp=1.1"""
        return self.translate_top_k(sentence, max_length=max_length, k=100, p=0.95, temperature=1.1)

    def calculate_accuracy_metrics(self, predictions, references):
        """Calculate accuracy metrics"""
        exact_matches = 0
        word_level_correct = 0
        total_words = 0

        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

            pred_words = pred.strip().lower().split()
            ref_words = ref.strip().lower().split()

            for i in range(min(len(pred_words), len(ref_words))):
                if pred_words[i] == ref_words[i]:
                    word_level_correct += 1
            total_words += len(ref_words)

        exact_match_acc = (exact_matches / len(predictions)) * 100
        word_level_acc = (word_level_correct / total_words) * 100 if total_words > 0 else 0

        return exact_match_acc, word_level_acc

    def test_sample_sentences(self):
        """Test on sample sentences"""
        sample_sentences = [
            "Hyvää huomenta",        # Good morning
            "Kiitos paljon",         # Thank you very much
            "Nähdään myöhemmin",     # See you later
            "Mikä on nimesi?",       # What is your name?
            "Pidän kahvista",        # I like coffee
            "Sää on kaunis tänään",  # The weather is beautiful today
            "Voitko auttaa minua?",  # Can you help me?
            "Olen opiskelija",       # I am a student
        ]

        print("\n" + "="*60)
        print("🧪 SAMPLE SENTENCE TESTING")
        print("="*60)

        results = []
        for i, sentence in enumerate(sample_sentences):
            print(f"\n{i+1}. Finnish: {sentence}")

            try:
                greedy_trans = self.translate_greedy(sentence)
                conservative_trans = self.translate_conservative(sentence)
                balanced_trans = self.translate_balanced(sentence)
                creative_trans = self.translate_creative(sentence)
                beam_trans = self.translate_beam_search(sentence, beam_size=4)
                beam_alt_trans = self.translate_beam_search_alternative(sentence, beam_size=6)

                print(f"   Greedy:       {greedy_trans}")
                print(f"   Conservative: {conservative_trans}")
                print(f"   Balanced:     {balanced_trans}")
                print(f"   Creative:     {creative_trans}")
                print(f"   Beam:         {beam_trans}")
                print(f"   Beam-Alt:     {beam_alt_trans}")

                # Check for UNK tokens and warn
                if '<UNK>' in beam_trans:
                    print(f"   ⚠️  Warning: Beam search still producing UNK tokens!")
                if '<UNK>' in greedy_trans:
                    print(f"   ⚠️  Warning: Greedy search producing UNK tokens!")

                # Check for very short translations
                if len(beam_trans.split()) <= 1:
                    print(f"   ⚠️  Warning: Beam search producing very short translation!")
                if len(beam_alt_trans.split()) <= 1:
                    print(f"   ⚠️  Warning: Alternative beam search producing very short translation!")

                results.append({
                    'Finnish': sentence,
                    'Greedy': greedy_trans,
                    'Conservative': conservative_trans,
                    'Balanced': balanced_trans,
                    'Creative': creative_trans,
                    'Beam': beam_trans,
                    'Beam_Alternative': beam_alt_trans
                })

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'Finnish': sentence,
                    'Greedy': f"Error: {e}",
                    'Conservative': f"Error: {e}",
                    'Balanced': f"Error: {e}",
                    'Creative': f"Error: {e}",
                    'Beam': f"Error: {e}",
                    'Beam_Alternative': f"Error: {e}"
                })

        # Save results
        df = pd.DataFrame(results)
        df.to_csv('/kaggle/working/sample_translations.csv', index=False)
        print(f"\n💾 Sample translations saved to /kaggle/working/sample_translations.csv")

        # Debug vocabulary information
        print(f"\n🔍 VOCABULARY DEBUG INFO:")
        print(f"   UNK token: '{self.tgt_vocab.UNK_TOKEN}' (idx: {self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)})")
        print(f"   SOS token: '{self.tgt_vocab.SOS_TOKEN}' (idx: {self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)})")
        print(f"   EOS token: '{self.tgt_vocab.EOS_TOKEN}' (idx: {self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)})")
        print(f"   PAD token: '{self.tgt_vocab.PAD_TOKEN}' (idx: {self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)})")
        print(f"   Target vocab size: {len(self.tgt_vocab)}")

        return results

    def single_strategy_evaluation(self, strategy='greedy', use_full_test=True):
        """Single strategy evaluation for focused testing"""
        print(f"\n" + "="*80)
        print(f"🎯 SINGLE STRATEGY EVALUATION: {strategy.upper()}")
        print("="*80)

        # Load test data
        print("📁 Loading test data...")
        try:
            en_file = f"{DATA_PATH}/EUbookshop.en"
            fi_file = f"{DATA_PATH}/EUbookshop.fi"

            if not os.path.exists(en_file) or not os.path.exists(fi_file):
                print(f"❌ Data files not found:")
                print(f"   Expected: {en_file}")
                print(f"   Expected: {fi_file}")
                return None

            en_sentences, fi_sentences = load_data(en_file, fi_file)
            train_data, val_data, test_data = split_data(en_sentences, fi_sentences, train_ratio=0.8, val_ratio=0.1)

            if use_full_test:
                test_src = test_data[1]  # Finnish
                test_tgt = test_data[0]  # English
                print(f"✅ Using full test set: {len(test_src):,} sentences")
            else:
                test_src = test_data[1][:100]
                test_tgt = test_data[0][:100]
                print(f"✅ Using test subset: {len(test_src)} sentences")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        print(f"\n📊 Test set size: {len(test_src)} sentence pairs")
        print(f"🤖 Model: {self.config['pos_encoding_type']} positional encoding")
        print(f"⚙️  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎭 Strategy: {strategy.upper()}")
        print("="*80)

        # Generate translations
        predictions = []
        references = test_tgt

        if strategy == 'greedy':
            print("⚡ Generating translations with Greedy decoding...")
            for src_sentence in tqdm(test_src, desc="Greedy Translation"):
                pred = self.translate_greedy(src_sentence)
                predictions.append(pred)
        elif strategy == 'conservative':
            print("🛡️  Generating translations with Conservative top-k...")
            for src_sentence in tqdm(test_src, desc="Conservative Translation"):
                pred = self.translate_conservative(src_sentence)
                predictions.append(pred)
        elif strategy == 'balanced':
            print("⚖️  Generating translations with Balanced top-k...")
            for src_sentence in tqdm(test_src, desc="Balanced Translation"):
                pred = self.translate_balanced(src_sentence)
                predictions.append(pred)
        elif strategy == 'creative':
            print("🎨 Generating translations with Creative top-k...")
            for src_sentence in tqdm(test_src, desc="Creative Translation"):
                pred = self.translate_creative(src_sentence)
                predictions.append(pred)
        elif strategy == 'beam':
            print("🔍 Generating translations with Beam search...")
            for src_sentence in tqdm(test_src, desc="Beam Translation"):
                pred = self.translate_beam_search(src_sentence, beam_size=4)
                predictions.append(pred)
        else:
            print(f"❌ Unknown strategy: {strategy}")
            return None

        # Show examples
        print(f"\n📝 Translation Examples (first 5):")
        for i in range(min(5, len(test_src))):
            print(f"\n--- Example {i+1} ---")
            print(f"Source (FI):  {test_src[i]}")
            print(f"Reference:    {references[i]}")
            print(f"{strategy.capitalize():12}: {predictions[i]}")

        # Calculate BLEU score
        print(f"\n📈 EVALUATION RESULTS")
        print("="*50)
        try:
            bleu_score = calculate_bleu(predictions, references)
            print(f"🎯 BLEU Score ({strategy}): {bleu_score:.4f}")
        except Exception as e:
            print(f"❌ Error calculating BLEU: {e}")
            bleu_score = 0.0

        # Calculate BERTScore
        print(f"\n🤖 Calculating BERTScore...")
        try:
            bert_scores = calculate_bert_score(predictions, references, device=self.device)
            print(f"📊 BERTScore Results:")
            print(f"   Precision: {bert_scores['precision_avg']:.4f}")
            print(f"   Recall:    {bert_scores['recall_avg']:.4f}")
            print(f"   F1:        {bert_scores['f1_avg']:.4f}")
        except Exception as e:
            print(f"❌ Error calculating BERTScore: {e}")
            bert_scores = {
                'precision_avg': 0.0,
                'recall_avg': 0.0,
                'f1_avg': 0.0,
                'precision_scores': [0.0] * len(predictions),
                'recall_scores': [0.0] * len(predictions),
                'f1_scores': [0.0] * len(predictions)
            }

        # Calculate additional metrics
        exact_match, word_acc = self.calculate_accuracy_metrics(predictions, references)
        avg_pred_len = sum(len(s.split()) for s in predictions) / len(predictions)
        avg_ref_len = sum(len(s.split()) for s in references) / len(references)

        print(f"\n📊 DETAILED METRICS:")
        print(f"  BLEU Score:      {bleu_score:.4f}")
        print(f"  BERTScore F1:    {bert_scores['f1_avg']:.4f}")
        print(f"  BERTScore Prec:  {bert_scores['precision_avg']:.4f}")
        print(f"  BERTScore Rec:   {bert_scores['recall_avg']:.4f}")
        print(f"  Exact Match:     {exact_match:.2f}%")
        print(f"  Word Accuracy:   {word_acc:.2f}%")
        print(f"  Avg Length:      {avg_pred_len:.1f} words (ref: {avg_ref_len:.1f})")

        # Save results
        results_df = pd.DataFrame({
            'Source_Finnish': test_src,
            'Reference_English': references,
            f'{strategy.capitalize()}_Translation': predictions
        })
        results_df.to_csv(f'/kaggle/working/{strategy}_test_results.csv', index=False)

        # Save summary
        summary = {
            'strategy': strategy,
            'test_sentences': len(test_src),
            'bleu_score': bleu_score,
            'bert_score_f1': bert_scores['f1_avg'],
            'bert_score_precision': bert_scores['precision_avg'],
            'bert_score_recall': bert_scores['recall_avg'],
            'exact_match_accuracy': exact_match,
            'word_accuracy': word_acc,
            'avg_prediction_length': avg_pred_len,
            'avg_reference_length': avg_ref_len,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'positional_encoding': self.config['pos_encoding_type'],
            'vocab_sizes': {
                'finnish': len(self.src_vocab),
                'english': len(self.tgt_vocab)
            }
        }

        with open(f'/kaggle/working/{strategy}_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"  📄 Detailed results: /kaggle/working/{strategy}_test_results.csv")
        print(f"  📊 Summary metrics: /kaggle/working/{strategy}_evaluation_summary.json")
        print("="*80)

        return summary

def main():
    """Main testing function"""

    # Check if modules were imported successfully
    if not modules_imported:
        print("❌ Cannot proceed without required modules")
        print("📋 To fix this:")
        print("1. Make sure encoder.py, decoder.py, and utils.py are uploaded to your Kaggle dataset")
        print("2. Check that the files are in the correct path")
        print("3. Ensure sacrebleu is installed (script will try to install it)")
        return
    print("🚀 Starting Transformer Testing on Kaggle")
    print("="*60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU not available - using CPU")
        print("   💡 In Kaggle, make sure to enable GPU in Settings > Accelerator")

    # Model paths to try - modify this to test specific checkpoints
    model_paths = [
        '/kaggle/working/best_model.pt',      # Best validation performance (recommended)
        '/kaggle/working/final_model.pt',     # Final training state
        '/kaggle/working/checkpoint_epoch_2.pt',  # Specific epoch
        f'{DATA_PATH}/best_model.pt',         # If model is in dataset
        f'{DATA_PATH}/rope_topk/best_model.pt',  # RoPE model from dataset
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ No model found! Please either:")
        print("   1. Run training first in the same session, or")
        print("   2. Upload your trained model to Kaggle input")
        return

    print(f"📥 Using model: {model_path}")

    try:
        # Initialize tester
        tester = ModelTester(model_path, device)

        # Test sample sentences
        print("\n" + "="*60)
        print("🧪 Testing sample sentences...")
        sample_results = tester.test_sample_sentences()

        # Single strategy evaluation for BLEU score
        print("\n" + "="*60)
        print("🎯 Running single strategy evaluation...")

        # You can modify these parameters:
        USE_FULL_TEST = True           # Set to False for faster testing with subset
        STRATEGY = 'greedy'          # Choose: 'greedy', 'conservative', 'balanced', 'creative', 'beam'

        print(f"Selected strategy: {STRATEGY}")

        summary = tester.single_strategy_evaluation(
            strategy=STRATEGY,
            use_full_test=USE_FULL_TEST
        )

        if summary:
            print("\n✅ Testing completed successfully!")
            print(f"🎯 BLEU Score ({STRATEGY}): {summary['bleu_score']:.4f}")
            print(f"📊 Test Sentences: {summary['test_sentences']:,}")
            print(f"🎭 Strategy: {STRATEGY.upper()}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Important: Update these paths for your Kaggle datasets
    print("🔧 Configuration:")
    print(f"   DATA_PATH: {DATA_PATH}")
    print("="*60)

    main()
