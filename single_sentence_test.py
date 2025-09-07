#!/usr/bin/env python3
"""
Single Sentence Translation Tester
Quick script to test individual Finnish sentences using your trained model
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add paths for imports (adjust for your environment)
DATA_PATH = "/kaggle/input/anlp-ass1"  # For Kaggle
# DATA_PATH = "."  # For local testing

sys.path.append(DATA_PATH)
sys.path.append("/kaggle/working")
sys.path.append(".")

# Import your modules
from encoder import TransformerEncoder
from decoder import TransformerDecoder, Transformer
from utils import (
    create_padding_mask, create_look_ahead_mask,
    indices_to_sentence, Vocabulary
)

class SingleSentenceTester:
    def __init__(self, model_path, device='auto'):
        """Initialize the tester with a trained model"""

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Using device: {self.device}")

        # Load checkpoint
        print(f"📥 Loading model from {model_path}")
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure the model file exists and is accessible")
            raise

        # Extract components
        self.src_vocab = self.checkpoint['src_vocab']  # Finnish
        self.tgt_vocab = self.checkpoint['tgt_vocab']  # English
        self.config = self.checkpoint['args']

        print(f"✅ Model loaded successfully!")
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
        """Translate using greedy decoding"""
        print(f"🔍 Greedy Decoding:")
        print(f"   Input: {sentence}")

        # Convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode
            encoder_output = self.model.encoder(src, src_mask)

            # Decode step by step
            tgt_indices = [self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]

            for step in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)

                # Get next token
                next_token_logits = decoder_output[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                tgt_indices.append(next_token)

                # Show intermediate steps
                if show_steps:
                    current_translation = indices_to_sentence(tgt_indices, self.tgt_vocab)
                    print(f"   Step {step+1}: {current_translation}")

                # Stop if EOS token
                if next_token == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                    break

        translation = indices_to_sentence(tgt_indices, self.tgt_vocab)
        print(f"   Output: {translation}")
        return translation

    def translate_beam_search(self, sentence, beam_size=4, max_length=50):
        """Translate using beam search"""
        print(f"🔍 Beam Search (beam_size={beam_size}):")
        print(f"   Input: {sentence}")

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
                    log_probs = F.log_softmax(decoder_output[0, -1, :], dim=-1)

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

        # Return best beam
        translation = indices_to_sentence(beams[0], self.tgt_vocab)
        print(f"   Output: {translation}")
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
            print(f"   Greedy: {results['greedy']}")
            print(f"   Beam:   {results['beam']}")

            if results['greedy'] == results['beam']:
                print("   ✅ Both methods agree!")
            else:
                print("   ⚖️  Different results - beam search might be better")

        print("="*60)
        return results

def main():
    """Main function for interactive testing"""

    # Model path - adjust this to your model location
    model_paths = [
        '/kaggle/working/best_model.pt',      # Kaggle
        './best_model.pt',                    # Local
        'best_model.pt'                       # Current directory
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ No model found! Please check these locations:")
        for path in model_paths:
            print(f"   {path}")
        return

    # Initialize tester
    try:
        tester = SingleSentenceTester(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return

    # Test sentences
    test_sentences = [
        "Hyvää huomenta",           # Good morning
        "Kiitos paljon",            # Thank you very much
        "Mikä on nimesi?",          # What is your name?
        "Pidän kahvista",           # I like coffee
        "Sää on kaunis tänään",     # The weather is beautiful today
        "Voitko auttaa minua?",     # Can you help me?
        "Olen opiskelija",          # I am a student
        "Nähdään myöhemmin"         # See you later
    ]

    print("🧪 Testing sample sentences...")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Test {i} ---")
        tester.translate(sentence, method='both')
        if i < len(test_sentences):
            input("Press Enter for next sentence...")

    # Interactive mode
    print("\n🎮 Interactive Mode (type 'quit' to exit)")
    print("Enter Finnish sentences to translate:")

    while True:
        try:
            sentence = input("\nFinnish: ").strip()
            if sentence.lower() in ['quit', 'exit', 'q']:
                break
            if sentence:
                tester.translate(sentence, method='both')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
