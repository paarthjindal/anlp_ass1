#!/usr/bin/env python3
"""
Local Single Sentence Translation Tester
Quick script to test individual Finnish sentences using your trained model locally
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add current directory to path for imports
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules (make sure encoder.py, decoder.py, utils.py are in same folder)
try:
    from encoder import TransformerEncoder
    from decoder import TransformerDecoder, Transformer
    from utils import (
        create_padding_mask, create_look_ahead_mask,
        indices_to_sentence, Vocabulary
    )
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("💡 Make sure encoder.py, decoder.py, and utils.py are in the same folder as this script")
    sys.exit(1)

class LocalSentenceTester:
    def __init__(self, model_path, device='auto'):
        """Initialize the tester with a trained model"""

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   CPU mode (slower but works everywhere)")

        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Make sure you've downloaded the .pt file to the correct location")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        print(f"📥 Loading model from {model_path}")
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure the model file is not corrupted and was trained with compatible PyTorch version")
            raise

        # Extract components
        self.src_vocab = self.checkpoint['src_vocab']  # Finnish
        self.tgt_vocab = self.checkpoint['tgt_vocab']  # English
        self.config = self.checkpoint['args']

        print(f"📊 Source vocab size: {len(self.src_vocab):,} (Finnish)")
        print(f"📊 Target vocab size: {len(self.tgt_vocab):,} (English)")

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

        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🤖 Model parameters: {total_params:,}")
        print(f"🔧 Positional encoding: {self.config['pos_encoding_type']}")
        print("="*60)

    def top_k_sampling_decode(self, logits, previous_tokens, step, k=50, p=0.9, temperature=1.0):
        """
        Clean top-k and nucleus sampling without word-specific tricks

        Args:
            logits: Raw model logits for next token prediction
            previous_tokens: List of previously generated tokens
            step: Current generation step
            k: Number of top tokens to consider
            p: Nucleus probability threshold
            temperature: Temperature for softmax
        """
        # Get special token indices
        unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
        eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
        pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)

        # Apply temperature
        logits = logits / temperature

        # Remove invalid tokens (UNK, PAD)
        logits[unk_idx] = float('-inf')
        logits[pad_idx] = float('-inf')

        # Basic repetition prevention (only immediate repetition)
        if len(previous_tokens) > 0:
            last_token = previous_tokens[-1]
            if last_token not in [eos_idx, pad_idx, unk_idx]:
                logits[last_token] -= 2.0  # Mild penalty for immediate repetition

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-k filtering
        if k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(k, probs.size(-1)))
            # Zero out probabilities outside top-k
            filtered_probs = torch.zeros_like(probs)
            filtered_probs.scatter_(-1, top_k_indices, top_k_probs)
            probs = filtered_probs

        # Nucleus (top-p) sampling
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
        if probs.sum() > 1e-6:
            next_token = torch.multinomial(probs, 1).item()
        else:
            # Fallback to argmax if all probabilities are zero
            logits[unk_idx] = float('-inf')
            logits[pad_idx] = float('-inf')
            next_token = torch.argmax(logits).item()

        return next_token

    def sentence_to_indices(self, sentence, vocab, max_length=None):
        """Convert sentence to indices"""
        if max_length is None:
            max_length = self.config['max_seq_len']

        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        # Truncate if too long
        if len(indices) > max_length - 2:
            indices = indices[:max_length - 2]

        # Add SOS and EOS tokens
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad to max length
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    def translate_greedy(self, sentence, max_length=50, show_steps=False):
        """Translate using improved Top-K sampling (better than pure greedy)"""
        print(f"🔍 Improved Top-K Sampling:")
        print(f"   Input: '{sentence}'")

        # Convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode
            encoder_output = self.model.encoder(src, src_mask)

            # Decode step by step with improved sampling
            tgt_indices = [self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]

            for step in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)

                # Get next token using improved Top-K sampling
                next_token_logits = decoder_output[0, -1, :]

                # Use conservative parameters for "greedy" to get best single result
                next_token = self.top_k_sampling_decode(next_token_logits, tgt_indices, step, k=15, p=0.8, temperature=0.8)
                tgt_indices.append(next_token)

                # Show intermediate steps
                if show_steps:
                    current_translation = indices_to_sentence(tgt_indices, self.tgt_vocab)
                    print(f"   Step {step+1}: '{current_translation}'")

                # Stop if EOS token
                if next_token == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                    break

        translation = indices_to_sentence(tgt_indices, self.tgt_vocab)
        print(f"   Output: '{translation}'")
        return translation

    def translate_beam_search(self, sentence, beam_size=4, max_length=50):
        """Translate using beam search"""
        print(f"🔍 Beam Search (beam_size={beam_size}):")
        print(f"   Input: '{sentence}'")

        # Convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode
            encoder_output = self.model.encoder(src, src_mask)

            # Initialize beams
            beams = [[self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]]
            beam_scores = [0.0]

            for step in range(max_length):
                candidates = []

                for i, beam in enumerate(beams):
                    # Skip if beam already ended
                    if beam[-1] == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                        candidates.append((beam_scores[i], beam))
                        continue

                    # Get predictions for this beam
                    tgt = torch.tensor([beam], device=self.device)
                    tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                    decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                    logits = decoder_output[0, -1, :]

                    # IMPORTANT: Filter out UNK and PAD tokens before beam search
                    unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)
                    pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)
                    eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)

                    logits[unk_idx] = float('-inf')
                    logits[pad_idx] = float('-inf')

                    # Prevent immediate repetition in beam search too
                    if len(beam) > 1:
                        last_token = beam[-1]
                        if last_token not in [eos_idx, pad_idx, unk_idx]:
                            logits[last_token] -= 2.0

                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get top candidates
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)

                    for j in range(beam_size):
                        new_beam = beam + [top_k_indices[j].item()]
                        new_score = beam_scores[i] + top_k_probs[j].item()
                        candidates.append((new_score, new_beam))

                # Keep best beams
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = [cand[1] for cand in candidates[:beam_size]]
                beam_scores = [cand[0] for cand in candidates[:beam_size]]

                # Check if all beams ended
                if all(beam[-1] == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN) for beam in beams):
                    break

        # Return best beam with length normalization
        if beams:
            # Normalize scores by length to prevent bias toward short sequences
            normalized_scores = []
            for i, beam in enumerate(beams):
                length = len([t for t in beam if t not in [
                    self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN),
                    self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN),
                    self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)
                ]])
                # Length normalization with slight bias toward longer sequences
                normalized_score = beam_scores[i] / (length + 1e-6) if length > 0 else beam_scores[i]
                normalized_scores.append((normalized_score, beam))

            # Pick best normalized beam
            best_beam = max(normalized_scores, key=lambda x: x[0])[1]
            translation = indices_to_sentence(best_beam, self.tgt_vocab)
        else:
            translation = "Error: No valid beam found"
        print(f"   Output: '{translation}'")
        return translation

    def translate(self, sentence, method='both', beam_size=4, show_steps=False):
        """Translate a sentence using specified method(s)"""
        print(f"\n🌍 Translating Finnish → English")
        print("="*60)

        results = {}

        if method in ['greedy', 'both']:
            results['greedy'] = self.translate_greedy(sentence, show_steps=show_steps)
            print()

        if method in ['beam', 'both']:
            results['beam'] = self.translate_beam_search(sentence, beam_size=beam_size)
            print()

        if method == 'both':
            print("📊 Comparison:")
            print(f"   Top-K:  '{results['greedy']}'")
            print(f"   Beam:   '{results['beam']}'")

            if results['greedy'] == results['beam']:
                print("   ✅ Both methods agree!")
            else:
                print("   ⚖️  Different results - try both and pick the better one")

        print("="*60)
        return results

def find_model_file():
    """Find the model file in common locations"""
    possible_paths = [
        './best_model.pt',
        './final_model.pt',
        './model.pt',
        './checkpoint_epoch_1.pt',
        './checkpoint_epoch_2.pt',
        './checkpoint_epoch_3.pt',
        './checkpoint_epoch_4.pt',
        './checkpoint_epoch_5.pt',
        '../best_model.pt',
        os.path.expanduser('~/Downloads/best_model.pt'),
        os.path.expanduser('~/Desktop/best_model.pt')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def main():
    """Main function for interactive testing"""

    print("🏠 Local Finnish-English Translator")
    print("="*60)

    # Try to find model automatically
    model_path = find_model_file()

    if model_path is None:
        print("❌ No model file found automatically.")
        print("📁 Please enter the path to your .pt model file:")
        model_path = input("Model path: ").strip()

        if not model_path or not os.path.exists(model_path):
            print("❌ Invalid path. Please make sure the file exists.")
            return
    else:
        print(f"✅ Found model: {model_path}")

    # Initialize tester
    try:
        tester = LocalSentenceTester(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return

    # Interactive mode
    print("\n🎮 Interactive Finnish Translator")
    print("Type Finnish sentences to translate to English")
    print("Commands: 'quit'/'exit'/'q' to quit, 'help' for options")
    print("-" * 60)

    while True:
        try:
            sentence = input("\n🇫🇮 Finnish: ").strip()

            if sentence.lower() in ['quit', 'exit', 'q']:
                break
            elif sentence.lower() == 'help':
                print("\n📚 Available commands:")
                print("  - Type any Finnish sentence to translate")
                print("  - 'quit', 'exit', 'q' - Exit the program")
                print("  - 'help' - Show this help")
                print("\n💡 Example sentences to try:")
                print("  - Hyvää huomenta (Good morning)")
                print("  - Kiitos paljon (Thank you very much)")
                print("  - Mikä on nimesi? (What is your name?)")
                print("  - Pidän kahvista (I like coffee)")
                continue
            elif not sentence:
                continue

            # Translate the sentence
            result = tester.translate(sentence, method='both')

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Thanks for using the translator! Goodbye!")

def quick_test():
    """Quick test function with predefined sentences"""
    model_path = find_model_file()

    if model_path is None:
        print("❌ No model found for quick test")
        return

    try:
        tester = LocalSentenceTester(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return

    # Test sentences
    test_sentences = [
        "Hyvää huomenta",           # Good morning
        "Kiitos paljon",            # Thank you very much
        "Mikä on nimesi?",          # What is your name?
        "Pidän kahvista",           # I like coffee
        "Nähdään myöhemmin"         # See you later
    ]

    print("🧪 Quick Test Mode")
    print("="*60)

    for sentence in test_sentences:
        tester.translate(sentence, method='greedy')
        print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Local Finnish-English Translator')
    parser.add_argument('--model', '-m', type=str, help='Path to model file (.pt)')
    parser.add_argument('--sentence', '-s', type=str, help='Single sentence to translate')
    parser.add_argument('--method', choices=['greedy', 'beam', 'both'], default='both',
                        help='Translation method')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with sample sentences')

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    elif args.sentence:
        # Single sentence mode
        model_path = args.model or find_model_file()
        if model_path is None:
            print("❌ No model found. Use --model to specify path.")
            sys.exit(1)

        try:
            tester = LocalSentenceTester(model_path)
            tester.translate(args.sentence, method=args.method)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Interactive mode
        main()
