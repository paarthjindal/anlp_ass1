#!/usr/bin/env python3
"""
Model Testing Script for Transformer Machine Translation in Google Colab
This script loads a trained model and provides various testing functionalities
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from encoder import TransformerEncoder
from decoder import TransformerDecoder, Transformer
from utils import (
    load_data, split_data, create_vocabulary, TransformerDataset,
    create_padding_mask, create_look_ahead_mask, calculate_bleu,
    indices_to_sentence, Vocabulary
)

# Add safe globals for PyTorch loading
try:
    torch.serialization.add_safe_globals([Vocabulary])
except Exception:
    pass  # Ignore if already added or not available

class ModelTester:
    def __init__(self, model_path, device='cuda'):
        """Initialize the model tester"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # Load checkpoint
        print(f"Loading model from {model_path}")

        # Handle PyTorch's weights_only security feature for custom classes
        try:
            # Try loading with weights_only=False for custom classes like Vocabulary
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading with weights_only=False: {e}")
            # Fallback: Add safe globals for custom classes
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

        # Initialize model
        self.model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            d_ff=self.config['d_ff'],
            max_seq_len=self.config['max_seq_len'],
            dropout=self.config['dropout'],
            pos_encoding_type=self.config['pos_encoding_type']
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Positional encoding: {self.config['pos_encoding_type']}")
        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def sentence_to_indices(self, sentence, vocab, max_length=None):
        """Convert sentence to indices"""
        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        if max_length is None:
            max_length = self.config['max_seq_len']

        if len(indices) > max_length - 2:  # -2 for SOS and EOS
            indices = indices[:max_length - 2]

        # Add SOS and EOS
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad if necessary
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    def translate_greedy(self, sentence, max_length=50):
        """Translate a sentence using greedy decoding - optimized version"""
        self.model.eval()

        # Tokenize and convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)

        # Create source mask
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode source
            encoder_output = self.model.encoder(src, src_mask)

            # Start with SOS token
            tgt_indices = [self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]

            for _ in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                # Decode
                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)

                # Get next token
                next_token_logits = decoder_output[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                tgt_indices.append(next_token)

                # Stop if EOS token
                if next_token == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                    break

        # Convert back to sentence
        return indices_to_sentence(tgt_indices, self.tgt_vocab)

    def translate_beam_search(self, sentence, beam_size=4, max_length=50):
        """Translate a sentence using beam search - optimized version"""
        self.model.eval()

        # Tokenize and convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)

        # Create source mask
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode source
            encoder_output = self.model.encoder(src, src_mask)

            # Initialize beams
            beams = [[self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]]
            beam_scores = [0.0]

            for step in range(max_length):
                candidates = []

                for i, beam in enumerate(beams):
                    if beam[-1] == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                        candidates.append((beam_scores[i], beam))
                        continue

                    tgt = torch.tensor([beam], device=self.device)
                    tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                    # Decode
                    decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)

                    # Get probabilities for next token
                    next_token_logits = decoder_output[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)

                    # Get top-k tokens
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)

                    for j in range(beam_size):
                        new_beam = beam + [top_k_indices[j].item()]
                        new_score = beam_scores[i] + top_k_probs[j].item()
                        candidates.append((new_score, new_beam))

                # Select top beams
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = [cand[1] for cand in candidates[:beam_size]]
                beam_scores = [cand[0] for cand in candidates[:beam_size]]

                # Check if all beams ended
                if all(beam[-1] == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN) for beam in beams):
                    break

        # Return best beam
        best_beam = beams[0]
        return indices_to_sentence(best_beam, self.tgt_vocab)

    def get_attention_weights(self, sentence):
        """Get attention weights for visualization"""
        self.model.eval()

        # Tokenize and convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices]).to(self.device)

        # Create source mask
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        # Get a simple translation first
        translation = self.translate_greedy(sentence)
        tgt_indices = self.sentence_to_indices(translation, self.tgt_vocab)
        tgt = torch.tensor([tgt_indices]).to(self.device)
        tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

        # Forward pass to get attention weights
        with torch.no_grad():
            # This is a simplified version - you might need to modify based on your model structure
            encoder_output = self.model.encoder(src, src_mask)
            decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return translation

    def calculate_accuracy_metrics(self, predictions, references):
        """Calculate various accuracy metrics"""
        exact_matches = 0
        word_level_correct = 0
        total_words = 0

        for pred, ref in zip(predictions, references):
            # Exact match accuracy
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

            # Word-level accuracy
            pred_words = pred.strip().lower().split()
            ref_words = ref.strip().lower().split()

            for i in range(min(len(pred_words), len(ref_words))):
                if pred_words[i] == ref_words[i]:
                    word_level_correct += 1
            total_words += len(ref_words)

        exact_match_acc = (exact_matches / len(predictions)) * 100
        word_level_acc = (word_level_correct / total_words) * 100 if total_words > 0 else 0

        return exact_match_acc, word_level_acc

    def evaluate_on_test_set(self, test_data_path=None, use_full_test=True, greedy_only=False):
        """Comprehensive evaluation on test set"""
        if test_data_path is None:
            # Use the data from training if available
            print("Loading test data...")

            # Try multiple possible data locations
            possible_paths = [
                './EUbookshop',
                '/content/EUbookshop',
                '/content/drive/MyDrive/EUbookshop',
                './data/EUbookshop',
                '/content/drive/MyDrive/data/EUbookshop'
            ]

            data_dir = None
            for path in possible_paths:
                data_dir = Path(path)
                src_file = data_dir / 'EUbookshop.fi'
                tgt_file = data_dir / 'EUbookshop.en'

                if src_file.exists() and tgt_file.exists():
                    print(f"Found data files in: {data_dir}")
                    break
            else:
                print("Test data files not found in any of the expected locations:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("Please ensure EUbookshop.fi and EUbookshop.en are available.")
                return

            # Load and split data (same as training: 80% train, 10% val, 10% test)
            en_sentences, fi_sentences = load_data(str(tgt_file), str(src_file))
            train_data, val_data, test_data = split_data(en_sentences, fi_sentences, train_ratio=0.8, val_ratio=0.1)

            if use_full_test:
                test_src = test_data[1]  # All Finnish test sentences (10% of data)
                test_tgt = test_data[0]  # All English test sentences (10% of data)
                print(f"Using full test set: {len(test_src)} sentences (10% of total data)")
            else:
                test_src = test_data[1][:100]  # First 100 Finnish sentences from test set
                test_tgt = test_data[0][:100]  # First 100 English sentences from test set
                print(f"Using test subset: {len(test_src)} sentences from the 10% test split")

        predictions_greedy = []
        predictions_beam = []
        references = []

        print("="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        print(f"Test set size: {len(test_src)} sentence pairs")
        print(f"Model: {self.config['pos_encoding_type']} positional encoding")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if greedy_only:
            print("Mode: FAST (Greedy decoding only)")
        else:
            print("Mode: FULL (Greedy + Beam search)")
        print("="*80)

        if greedy_only:
            # Fast mode: Only greedy decoding
            print("\nGenerating translations with greedy decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Greedy")):
                pred_greedy = self.translate_greedy(src_sentence)
                predictions_greedy.append(pred_greedy)
                references.append(test_tgt[i])

                # Show first 10 examples
                if i < 10:
                    print(f"\nExample {i+1}:")
                    print(f"Source (FI): {src_sentence}")
                    print(f"Reference (EN): {test_tgt[i]}")
                    print(f"Greedy (EN): {pred_greedy}")

            # No beam search in fast mode
            predictions_beam = predictions_greedy.copy()  # Use greedy results

        else:
            # Full mode: Both greedy and beam search
            print("\nGenerating translations with both decoding strategies...")

            # First pass: Greedy decoding (faster)
            print("Phase 1/2: Greedy decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Greedy")):
                pred_greedy = self.translate_greedy(src_sentence)
                predictions_greedy.append(pred_greedy)
                references.append(test_tgt[i])

            # Second pass: Beam search decoding
            print("Phase 2/2: Beam search decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Beam")):
                pred_beam = self.translate_beam_search(src_sentence, beam_size=4)
                predictions_beam.append(pred_beam)

                # Show first 10 examples during beam search phase
                if i < 10:
                    print(f"\nExample {i+1}:")
                    print(f"Source (FI): {src_sentence}")
                    print(f"Reference (EN): {test_tgt[i]}")
                    print(f"Greedy (EN): {predictions_greedy[i]}")
                    print(f"Beam (EN): {pred_beam}")

        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        # Calculate BLEU scores
        try:
            bleu_greedy = calculate_bleu(predictions_greedy, references)
            bleu_beam = calculate_bleu(predictions_beam, references)

            print(f"\n📊 BLEU SCORES:")
            print(f"  Greedy Decoding: {bleu_greedy:.4f}")
            print(f"  Beam Search:     {bleu_beam:.4f}")
            print(f"  Improvement:     {bleu_beam - bleu_greedy:+.4f}")

        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            bleu_greedy = bleu_beam = 0

        # Calculate accuracy metrics
        exact_match_greedy, word_acc_greedy = self.calculate_accuracy_metrics(predictions_greedy, references)
        exact_match_beam, word_acc_beam = self.calculate_accuracy_metrics(predictions_beam, references)

        print(f"\n🎯 ACCURACY METRICS:")
        print(f"  Exact Match Accuracy:")
        print(f"    Greedy:  {exact_match_greedy:.2f}%")
        print(f"    Beam:    {exact_match_beam:.2f}%")
        print(f"  Word-level Accuracy:")
        print(f"    Greedy:  {word_acc_greedy:.2f}%")
        print(f"    Beam:    {word_acc_beam:.2f}%")

        # Calculate average sentence lengths
        avg_src_len = sum(len(s.split()) for s in test_src) / len(test_src)
        avg_ref_len = sum(len(s.split()) for s in references) / len(references)
        avg_pred_greedy_len = sum(len(s.split()) for s in predictions_greedy) / len(predictions_greedy)
        avg_pred_beam_len = sum(len(s.split()) for s in predictions_beam) / len(predictions_beam)

        print(f"\n📏 AVERAGE SENTENCE LENGTHS:")
        print(f"  Source (Finnish):    {avg_src_len:.1f} words")
        print(f"  Reference (English): {avg_ref_len:.1f} words")
        print(f"  Greedy Predictions:  {avg_pred_greedy_len:.1f} words")
        print(f"  Beam Predictions:    {avg_pred_beam_len:.1f} words")

        # Model performance summary
        print(f"\n🏆 MODEL PERFORMANCE SUMMARY:")
        print(f"  Dataset: Finnish → English")
        print(f"  Test sentences: {len(test_src):,}")
        print(f"  Best BLEU: {max(bleu_greedy, bleu_beam):.4f} ({'Beam' if bleu_beam > bleu_greedy else 'Greedy'})")
        print(f"  Best Accuracy: {max(exact_match_greedy, exact_match_beam):.2f}% ({'Beam' if exact_match_beam > exact_match_greedy else 'Greedy'})")
        print(f"  Vocabulary sizes: FI={len(self.src_vocab):,}, EN={len(self.tgt_vocab):,}")

        print("="*80)

        return {
            'predictions_greedy': predictions_greedy,
            'predictions_beam': predictions_beam,
            'references': references,
            'bleu_greedy': bleu_greedy,
            'bleu_beam': bleu_beam,
            'exact_match_greedy': exact_match_greedy,
            'exact_match_beam': exact_match_beam,
            'word_acc_greedy': word_acc_greedy,
            'word_acc_beam': word_acc_beam
        }

    def interactive_translation(self):
        """Interactive translation interface"""
        print("\n" + "="*50)
        print("Interactive Translation Interface")
        print("Type 'quit' to exit")
        print("="*50)

        while True:
            sentence = input("\nEnter Finnish sentence: ").strip()

            if sentence.lower() == 'quit':
                break

            if not sentence:
                continue

            try:
                # Greedy translation
                greedy_translation = self.translate_greedy(sentence)
                print(f"Greedy Translation: {greedy_translation}")

                # Beam search translation
                beam_translation = self.translate_beam_search(sentence, beam_size=4)
                print(f"Beam Search Translation: {beam_translation}")

            except Exception as e:
                print(f"Error during translation: {e}")

    def test_sample_sentences(self):
        """Test on predefined sample sentences"""
        sample_sentences = [
            "Hyvää huomenta",  # Good morning
            "Kiitos paljon",   # Thank you very much
            "Nähdään myöhemmin",  # See you later
            "Mikä on nimesi?",  # What is your name?
            "Pidän kahvista",  # I like coffee
            "Sää on kaunis tänään",  # The weather is beautiful today
            "Voitko auttaa minua?",  # Can you help me?
            "Olen opiskelija",  # I am a student
        ]

        print("\n" + "="*60)
        print("Testing Sample Sentences")
        print("="*60)

        for i, sentence in enumerate(sample_sentences):
            print(f"\n{i+1}. Finnish: {sentence}")

            try:
                greedy_trans = self.translate_greedy(sentence)
                beam_trans = self.translate_beam_search(sentence, beam_size=4)

                print(f"   Greedy: {greedy_trans}")
                print(f"   Beam:   {beam_trans}")

            except Exception as e:
                print(f"   Error: {e}")

def main():
    # Prevent multiple executions
    if os.environ.get('SCRIPT_RUNNING') == 'true':
        print("Script is already running. Preventing duplicate execution.")
        return

    os.environ['SCRIPT_RUNNING'] = 'true'

    try:
        # Add debug info
        print(f"Script called with args: {sys.argv}")

        parser = argparse.ArgumentParser(description='Test Transformer Model in Colab')
        parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                            help='Path to the trained model')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use for testing')
        parser.add_argument('--test_mode', type=str, default='samples',
                            choices=['samples', 'interactive', 'evaluate', 'comprehensive', 'comprehensive_fast', 'all'],
                            help='Testing mode')
        parser.add_argument('--full_test', action='store_true',
                            help='Use full test split instead of subset')
        parser.add_argument('--greedy_only', action='store_true',
                            help='Use only greedy decoding for faster testing')

        try:
            args = parser.parse_args()
            print(f"Parsed args: {args}")
        except SystemExit as e:
            # Code 0 means help was requested, code 2 means argument error
            if e.code == 0:
                print("Help was requested or normal exit")
                return
            else:
                print(f"Argument parsing failed with code: {e.code}")
                return
        except Exception as e:
            print(f"Unexpected error in argument parsing: {e}")
            return

        # Check if running in Colab
        try:
            import google.colab
            print("Running in Google Colab")
            if not os.path.exists('/content/drive'):
                from google.colab import drive
                drive.mount('/content/drive')
        except ImportError:
            print("Not running in Google Colab")

        # Check GPU
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
            args.device = 'cpu'

        # Initialize tester
        try:
            tester = ModelTester(args.model_path, args.device)
        except FileNotFoundError:
            print(f"Model file not found: {args.model_path}")
            print("Please ensure the model file exists or update the path.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Run tests based on mode
        if args.test_mode == 'samples' or args.test_mode == 'all':
            tester.test_sample_sentences()

        if args.test_mode == 'interactive' or args.test_mode == 'all':
            tester.interactive_translation()

        if args.test_mode == 'evaluate' or args.test_mode == 'all':
            tester.evaluate_on_test_set(use_full_test=args.full_test, greedy_only=args.greedy_only)

        if args.test_mode == 'comprehensive':
            tester.evaluate_on_test_set(use_full_test=True, greedy_only=args.greedy_only)

        if args.test_mode == 'comprehensive_fast':
            tester.evaluate_on_test_set(use_full_test=True, greedy_only=True)

    finally:
        os.environ['SCRIPT_RUNNING'] = 'false'

if __name__ == "__main__":
    main()
