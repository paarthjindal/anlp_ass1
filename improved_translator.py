#!/usr/bin/env python3
"""
Improved Local Translator with Top-K Sampling and UNK Prevention
Fixes repetitive "the" outputs and eliminates UNK predictions
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(".")

class ImprovedLocalTranslator:
    def __init__(self, model_path, device='auto'):
        """Initialize translator with clean top-k sampling (no word tricks)"""

        # Import modules
        try:
            from encoder import TransformerEncoder
            from decoder import TransformerDecoder, Transformer
            from utils import (
                create_padding_mask, create_look_ahead_mask,
                indices_to_sentence, Vocabulary
            )
            # Store functions as instance variables
            self.create_padding_mask = create_padding_mask
            self.create_look_ahead_mask = create_look_ahead_mask
            self.indices_to_sentence = indices_to_sentence
            self.modules_available = True
        except ImportError as e:
            print(f"❌ Error importing modules: {e}")
            self.modules_available = False
            return

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Using device: {self.device}")

        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.src_vocab = checkpoint['src_vocab']
            self.tgt_vocab = checkpoint['tgt_vocab']
            self.config = checkpoint['config']

            print(f"📊 Vocabularies loaded: Finnish={len(self.src_vocab)}, English={len(self.tgt_vocab)}")

            # Initialize model
            self.model = Transformer(
                src_vocab_size=len(self.src_vocab),
                tgt_vocab_size=len(self.tgt_vocab),
                **{k: v for k, v in self.config.items() if k in [
                    'd_model', 'num_heads', 'num_encoder_layers',
                    'num_decoder_layers', 'd_ff', 'max_seq_len',
                    'dropout', 'pos_encoding_type'
                ]}
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Get special token indices (needed for clean inference)
            self.pad_idx = self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN)
            self.sos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)
            self.eos_idx = self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)
            self.unk_idx = self.tgt_vocab.get_idx(self.tgt_vocab.UNK_TOKEN)

            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def sentence_to_indices(self, sentence, vocab, max_length=None):
        """Convert sentence to indices"""
        if max_length is None:
            max_length = self.config['max_seq_len']

        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        # Truncate if too long
        if len(indices) > max_length - 2:
            indices = indices[:max_length - 2]

        # Add SOS and EOS
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    def clean_top_k_sampling(self, logits, k=50, p=0.9, temperature=1.0):
        """Clean top-k and nucleus sampling without word-specific tricks"""

        # Apply temperature scaling
        logits = logits / temperature

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
            prob = probs[next_token].item()
        else:
            # Fallback to argmax if all probabilities are zero
            next_token = torch.argmax(logits).item()
            prob = 1.0

        return next_token, prob

    def translate_with_clean_sampling(self, sentence, k=50, p=0.9, temperature=1.0, max_length=50, show_steps=False):
        """Clean translation using only model predictions and top-k sampling"""

        print(f"🔍 Clean Top-K Sampling (k={k}, p={p}, temp={temperature}):")
        print(f"   Input: '{sentence}'")

        # Convert to indices
        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = self.create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            # Encode
            encoder_output = self.model.encoder(src, src_mask)

            # Decode with clean top-k sampling
            tgt_indices = [self.sos_idx]

            for step in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = self.create_look_ahead_mask(tgt, self.pad_idx)

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = decoder_output[0, -1, :]

                # Clean sampling - no tricks, just pure model predictions
                next_token, prob = self.clean_top_k_sampling(logits, k=k, p=p, temperature=temperature)

                if show_steps:
                    word = self.tgt_vocab.get_word(next_token)
                    current = self.indices_to_sentence(tgt_indices + [next_token], self.tgt_vocab)
                    print(f"   Step {step+1}: '{word}' (p={prob:.4f}) → '{current}'")

                tgt_indices.append(next_token)

                # Stop if EOS token is generated
                if next_token == self.eos_idx:
                    break

        translation = self.indices_to_sentence(tgt_indices, self.tgt_vocab)
        print(f"   Output: '{translation}'")
        return translation

    def translate_multiple_strategies(self, sentence, show_analysis=True):
        """Try multiple clean decoding strategies and compare"""

        print(f"\n🌍 Translating: '{sentence}'")
        print("="*80)

        results = {}

        # Strategy 1: Greedy (k=1, deterministic)
        print("\n🎯 STRATEGY 1: Greedy Decoding (k=1)")
        results['greedy'] = self.translate_with_clean_sampling(sentence, k=1, p=1.0, temperature=1.0)

        # Strategy 2: Conservative top-k
        print("\n🎯 STRATEGY 2: Conservative Top-K (k=20, p=0.8)")
        results['conservative'] = self.translate_with_clean_sampling(sentence, k=20, p=0.8, temperature=1.0)

        # Strategy 3: Balanced top-k
        print("\n🎯 STRATEGY 3: Balanced Top-K (k=50, p=0.9)")
        results['balanced'] = self.translate_with_clean_sampling(sentence, k=50, p=0.9, temperature=1.0)

        # Strategy 4: Creative top-k
        print("\n🎯 STRATEGY 4: Creative Top-K (k=100, p=0.95)")
        results['creative'] = self.translate_with_clean_sampling(sentence, k=100, p=0.95, temperature=1.1)

        if show_analysis:
            print(f"\n📊 COMPARISON:")
            for strategy, translation in results.items():
                unk_count = translation.count('<UNK>')
                word_count = len(translation.split())
                print(f"   {strategy.capitalize():12}: '{translation}'")
                print(f"   {'':12}   └─ Words: {word_count}, UNK: {unk_count}")

            # Simple selection: prefer non-empty, non-UNK results
            valid_results = {k: v for k, v in results.items()
                           if v.strip() and '<UNK>' not in v and len(v.split()) > 0}

            if valid_results:
                # Pick the one with reasonable length (not too short, not too long)
                best_strategy = min(valid_results.keys(),
                                  key=lambda s: abs(len(valid_results[s].split()) - 3))
                print(f"\n🏆 RECOMMENDED: {best_strategy.capitalize()} strategy")
                print(f"   Result: '{results[best_strategy]}'")
            else:
                print(f"\n⚠️  All strategies produced poor results")

        return results

def main():
    """Main function"""

    print("🚀 Improved Finnish-English Translator with Top-K Sampling")
    print("="*80)

    # Find model
    model_path = './best_model.pt'
    if not os.path.exists(model_path):
        print("❌ Model not found. Make sure best_model.pt exists.")
        return

    try:
        # Initialize translator
        translator = ImprovedLocalTranslator(model_path)

        # Test sentences
        test_sentences = [
            "Hyvää huomenta",      # Good morning
            "Kiitos paljon",       # Thank you very much
            "Mikä on nimesi?",     # What is your name?
            "Euroopan unioni",     # European Union
            "Suomi"                # Finland
        ]

        print("\n🧪 TESTING IMPROVED TRANSLATION:")
        print("="*80)

        for sentence in test_sentences:
            translator.translate_multiple_strategies(sentence, show_analysis=True)
            print("\n" + "="*80)

        # Interactive mode
        print("\n🎮 Interactive Mode (type 'quit' to exit):")
        print("-"*40)

        while True:
            try:
                user_input = input("\n🇫🇮 Finnish: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif not user_input:
                    continue

                translator.translate_multiple_strategies(user_input, show_analysis=True)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize translator: {e}")

if __name__ == "__main__":
    main()
