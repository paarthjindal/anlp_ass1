#!/usr/bin/env python3
"""
Probability Distribution Analysis
Analyze what probabilities your model assigns to different words
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(".")

def analyze_probability_distribution():
    """Analyze the probability distribution of model predictions"""

    try:
        from encoder import TransformerEncoder
        from decoder import TransformerDecoder, Transformer
        from utils import (
            create_padding_mask, create_look_ahead_mask,
            indices_to_sentence, Vocabulary
        )

        # Load model
        checkpoint = torch.load('./best_model.pt', map_location='cpu', weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        config = checkpoint['config']

        print(f"🔧 Model Info:")
        print(f"   Finnish vocab: {len(src_vocab)}")
        print(f"   English vocab: {len(tgt_vocab)}")
        print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")

        # Initialize model
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            **{k: v for k, v in config.items() if k in [
                'd_model', 'num_heads', 'num_encoder_layers',
                'num_decoder_layers', 'd_ff', 'max_seq_len',
                'dropout', 'pos_encoding_type'
            ]}
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Test sentence
        test_sentence = "Hyvää huomenta"  # "Good morning"
        print(f"\n🇫🇮 Analyzing: '{test_sentence}'")
        print("="*80)

        # Convert to indices
        words = test_sentence.lower().split()
        src_indices = [src_vocab.get_idx(src_vocab.SOS_TOKEN)]
        for word in words:
            src_indices.append(src_vocab.get_idx(word))
        src_indices.append(src_vocab.get_idx(src_vocab.EOS_TOKEN))

        # Pad to fixed length
        while len(src_indices) < 20:
            src_indices.append(src_vocab.get_idx(src_vocab.PAD_TOKEN))

        src = torch.tensor([src_indices])
        src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))

        with torch.no_grad():
            encoder_output = model.encoder(src, src_mask)

            # Start with SOS token
            tgt_indices = [tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)]

            print("📊 PROBABILITY ANALYSIS FOR EACH PREDICTION STEP:")
            print("-"*80)

            for step in range(5):  # Analyze first 5 prediction steps
                tgt = torch.tensor([tgt_indices])
                tgt_mask = create_look_ahead_mask(tgt, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

                decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = decoder_output[0, -1, :]  # Last token's predictions

                # Calculate probabilities
                probs = F.softmax(logits, dim=-1)

                print(f"\n🔍 STEP {step + 1}:")
                print(f"   Current sequence: {indices_to_sentence(tgt_indices, tgt_vocab)}")

                # Get top 15 predictions
                top_probs, top_indices = torch.topk(probs, 15)

                print(f"   📈 TOP 15 WORD PROBABILITIES:")
                print(f"   {'Rank':<4} {'Word':<15} {'Probability':<12} {'Logit':<10} {'Index':<6}")
                print(f"   {'-'*4:<4} {'-'*15:<15} {'-'*12:<12} {'-'*10:<10} {'-'*6:<6}")

                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    word = tgt_vocab.get_word(idx.item())
                    logit_val = logits[idx].item()
                    print(f"   {i+1:<4} {word:<15} {prob.item():<12.6f} {logit_val:<10.4f} {idx.item():<6}")

                # Analyze distribution statistics
                max_prob = torch.max(probs).item()
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                print(f"\n   📊 DISTRIBUTION STATISTICS:")
                print(f"   Max probability: {max_prob:.6f} ({max_prob*100:.2f}%)")
                print(f"   Entropy: {entropy:.4f} (lower = more confident)")
                print(f"   Effective vocab: {torch.sum(probs > 0.001).item()} words with >0.1% probability")

                # Check specific important words
                important_words = ["good", "morning", "hello", "hi", "the", "a", "and"]
                print(f"\n   🔍 PROBABILITIES FOR IMPORTANT WORDS:")
                for word in important_words:
                    word_idx = tgt_vocab.get_idx(word)
                    word_prob = probs[word_idx].item()
                    print(f"   '{word}': {word_prob:.6f} ({word_prob*100:.3f}%)")

                # Greedy selection
                next_token_idx = torch.argmax(probs).item()
                next_word = tgt_vocab.get_word(next_token_idx)
                next_prob = probs[next_token_idx].item()

                print(f"\n   🎯 GREEDY SELECTION:")
                print(f"   Chosen word: '{next_word}' (idx: {next_token_idx})")
                print(f"   Probability: {next_prob:.6f} ({next_prob*100:.2f}%)")

                # Add to sequence
                tgt_indices.append(next_token_idx)

                # Stop if EOS
                if next_token_idx == tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN):
                    print(f"   ⏹️  Stopped: EOS token predicted")
                    break

                print("="*80)

            print(f"\n🎯 FINAL TRANSLATION: '{indices_to_sentence(tgt_indices, tgt_vocab)}'")

            # Overall analysis
            print(f"\n💡 ANALYSIS SUMMARY:")
            print(f"   • Model validation loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
            if checkpoint.get('val_loss', 10) > 5.0:
                print(f"   • ❌ Very high validation loss indicates poor training")

            # Check if model is stuck in mode collapse
            print(f"\n🚨 MODE COLLAPSE ANALYSIS:")

            # Test multiple inputs to see if outputs are always the same
            test_inputs = ["Suomi", "Kiitos", "Hei", "Kyllä"]
            output_diversity = []

            for test_input in test_inputs:
                test_words = test_input.lower().split()
                test_src_indices = [src_vocab.get_idx(src_vocab.SOS_TOKEN)]
                for word in test_words:
                    test_src_indices.append(src_vocab.get_idx(word))
                test_src_indices.append(src_vocab.get_idx(src_vocab.EOS_TOKEN))

                while len(test_src_indices) < 20:
                    test_src_indices.append(src_vocab.get_idx(src_vocab.PAD_TOKEN))

                test_src = torch.tensor([test_src_indices])
                test_src_mask = create_padding_mask(test_src, src_vocab.get_idx(src_vocab.PAD_TOKEN))

                test_encoder_output = model.encoder(test_src, test_src_mask)
                test_tgt = torch.tensor([[tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)]])
                test_tgt_mask = create_look_ahead_mask(test_tgt, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

                test_decoder_output = model.decoder(test_tgt, test_encoder_output, test_src_mask, test_tgt_mask)
                test_probs = F.softmax(test_decoder_output[0, -1, :], dim=-1)
                test_top_idx = torch.argmax(test_probs).item()
                test_top_word = tgt_vocab.get_word(test_top_idx)

                output_diversity.append(test_top_word)
                print(f"   '{test_input}' → first predicted word: '{test_top_word}' ({test_probs[test_top_idx].item():.4f})")

            unique_outputs = len(set(output_diversity))
            print(f"\n   Unique first predictions: {unique_outputs}/{len(test_inputs)}")
            if unique_outputs <= 2:
                print(f"   ❌ SEVERE MODE COLLAPSE: Model predicts same words regardless of input")
            elif unique_outputs < len(test_inputs):
                print(f"   ⚠️  PARTIAL MODE COLLAPSE: Limited output diversity")
            else:
                print(f"   ✅ GOOD DIVERSITY: Different inputs produce different outputs")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_probability_distribution()
