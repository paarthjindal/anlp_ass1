#!/usr/bin/env python3
"""
Simple Translation Test with Fixed Issues
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(".")

def simple_test():
    """Simple test to isolate issues"""

    try:
        from encoder import TransformerEncoder
        from decoder import TransformerDecoder, Transformer
        from utils import (
            create_padding_mask, create_look_ahead_mask,
            indices_to_sentence, Vocabulary
        )

        # Load model
        model_path = input("Enter path to your .pt model file: ").strip()
        if not os.path.exists(model_path):
            print("File not found!")
            return

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        config = checkpoint['config']

        print(f"Vocab sizes: Finnish={len(src_vocab)}, English={len(tgt_vocab)}")

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
        model.eval()  # CRITICAL!

        # Test simple translation
        test_sentence = "Suomi"
        print(f"\nTranslating: '{test_sentence}'")

        # Convert to indices
        words = test_sentence.lower().split()
        src_indices = [src_vocab.get_idx(src_vocab.SOS_TOKEN)]
        for word in words:
            src_indices.append(src_vocab.get_idx(word))
        src_indices.append(src_vocab.get_idx(src_vocab.EOS_TOKEN))

        # Pad to reasonable length
        while len(src_indices) < 20:
            src_indices.append(src_vocab.get_idx(src_vocab.PAD_TOKEN))

        src = torch.tensor([src_indices])
        src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))

        with torch.no_grad():
            encoder_output = model.encoder(src, src_mask)

            # Simple greedy decoding with debugging
            tgt_indices = [tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)]

            for step in range(10):
                tgt = torch.tensor([tgt_indices])
                tgt_mask = create_look_ahead_mask(tgt, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

                decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = decoder_output[0, -1, :]

                # Add some randomness to avoid repetition
                probs = F.softmax(logits / 1.2, dim=-1)  # Temperature = 1.2

                # Get top-5 and sample
                top_probs, top_indices = torch.topk(probs, 5)

                # Sample from top-5
                next_token = top_indices[torch.multinomial(top_probs, 1)].item()
                word = tgt_vocab.get_word(next_token)

                print(f"Step {step+1}: {word}")

                tgt_indices.append(next_token)

                if next_token == tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN):
                    break

                # Break if repetition
                if len(tgt_indices) > 3 and tgt_indices[-1] == tgt_indices[-2] == tgt_indices[-3]:
                    print("Repetition detected, stopping")
                    break

        translation = indices_to_sentence(tgt_indices, tgt_vocab)
        print(f"\nFinal: '{translation}'")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
