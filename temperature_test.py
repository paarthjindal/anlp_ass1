#!/usr/bin/env python3
"""
Temperature-based Translation Fix
Try to get better results from current model using temperature sampling
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(".")

def test_with_temperature():
    """Test current model with temperature sampling to reduce repetition"""

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

        print(f"🔧 Model Info: Finnish={len(src_vocab)}, English={len(tgt_vocab)}")
        print(f"📊 Training Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")

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

        # Test sentences
        test_sentences = [
            "Suomi",
            "Hyvää huomenta",
            "Kiitos",
            "Euroopan unioni"
        ]

        temperatures = [0.8, 1.0, 1.2, 1.5]

        for sentence in test_sentences:
            print(f"\n🇫🇮 Finnish: '{sentence}'")
            print("🌍 Translations with different temperatures:")

            # Convert to indices
            words = sentence.lower().split()
            src_indices = [src_vocab.get_idx(src_vocab.SOS_TOKEN)]
            for word in words:
                src_indices.append(src_vocab.get_idx(word))
            src_indices.append(src_vocab.get_idx(src_vocab.EOS_TOKEN))

            # Pad
            while len(src_indices) < 20:
                src_indices.append(src_vocab.get_idx(src_vocab.PAD_TOKEN))

            src = torch.tensor([src_indices])
            src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))

            with torch.no_grad():
                encoder_output = model.encoder(src, src_mask)

                for temp in temperatures:
                    print(f"   🌡️  Temperature {temp}:")

                    tgt_indices = [tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)]

                    for step in range(15):
                        tgt = torch.tensor([tgt_indices])
                        tgt_mask = create_look_ahead_mask(tgt, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

                        decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                        logits = decoder_output[0, -1, :]

                        # Apply temperature
                        probs = F.softmax(logits / temp, dim=-1)

                        # Sample from top-k to avoid very rare words
                        top_k = 50
                        top_probs, top_indices = torch.topk(probs, top_k)

                        # Renormalize
                        top_probs = top_probs / top_probs.sum()

                        # Sample
                        sampled_idx = torch.multinomial(top_probs, 1)
                        next_token = top_indices[sampled_idx].item()

                        tgt_indices.append(next_token)

                        # Stop conditions
                        if next_token == tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN):
                            break

                        # Avoid infinite repetition
                        if len(tgt_indices) > 4:
                            last_3 = tgt_indices[-3:]
                            if len(set(last_3)) == 1:  # All same
                                break

                    translation = indices_to_sentence(tgt_indices, tgt_vocab)
                    print(f"      '{translation}'")

        print(f"\n💡 ANALYSIS:")
        print("   - Higher temperature = more random/creative")
        print("   - Lower temperature = more conservative")
        print("   - If all outputs are poor, model needs retraining")
        print("   - Try retraining with Cross Entropy loss (no label smoothing)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_temperature()
