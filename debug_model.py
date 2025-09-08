#!/usr/bin/env python3
"""
Model Debugging Script
Comprehensive diagnostics for translation issues
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add current directory to path
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_model_translation():
    """Debug model translation issues"""

    print("🔍 MODEL TRANSLATION DEBUGGER")
    print("="*60)

    # Import modules
    try:
        from encoder import TransformerEncoder
        from decoder import TransformerDecoder, Transformer
        from utils import (
            create_padding_mask, create_look_ahead_mask,
            indices_to_sentence, Vocabulary
        )
        print("✅ Successfully imported modules")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return

    # Find model file
    model_paths = [
        './best_model.pt',
        './final_model.pt',
        os.path.expanduser('~/Downloads/best_model.pt'),
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ No model file found!")
        print("Please ensure you have a trained model (.pt file)")
        return

    print(f"📁 Using model: {model_path}")

    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Extract components
    src_vocab = checkpoint['src_vocab']  # Finnish
    tgt_vocab = checkpoint['tgt_vocab']  # English
    config = checkpoint['config']

    print(f"\n📊 MODEL INFORMATION:")
    print(f"   Source vocab size: {len(src_vocab):,}")
    print(f"   Target vocab size: {len(tgt_vocab):,}")
    print(f"   Model architecture: {config['pos_encoding_type']}")
    print(f"   Training loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")

    # Check vocabulary health
    print(f"\n🔍 VOCABULARY ANALYSIS:")
    print(f"   Finnish vocab: {len(src_vocab)} words")
    print(f"   English vocab: {len(tgt_vocab)} words")

    if len(src_vocab) < 100 or len(tgt_vocab) < 100:
        print("   ❌ CRITICAL: Vocabulary too small!")
        print("   💡 This explains poor translation quality")
        print("   💡 You need to retrain with fixed vocabulary")
        return

    # Test sentence analysis
    test_sentence = "Suomi"
    print(f"\n🧪 INPUT ANALYSIS:")
    print(f"   Test sentence: '{test_sentence}'")

    words = test_sentence.lower().split()
    print(f"   Word breakdown:")
    for word in words:
        if word in src_vocab.word2idx:
            idx = src_vocab.get_idx(word)
            print(f"     '{word}' → idx {idx} ✅")
        else:
            print(f"     '{word}' → <UNK> ❌")

    # Initialize model
    device = torch.device('cpu')  # Use CPU for debugging

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pos_encoding_type=config['pos_encoding_type']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # CRITICAL: Set to evaluation mode

    print(f"\n🤖 MODEL STATUS:")
    print(f"   Mode: {'eval' if not model.training else 'train'}")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test translation step by step
    print(f"\n🔍 STEP-BY-STEP TRANSLATION:")

    # Convert sentence to indices
    def sentence_to_indices(sentence, vocab, max_length=128):
        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        if len(indices) > max_length - 2:
            indices = indices[:max_length - 2]

        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    # Test encoding
    src_indices = sentence_to_indices(test_sentence, src_vocab)
    src = torch.tensor([src_indices], device=device)
    src_mask = create_padding_mask(src, src_vocab.get_idx(src_vocab.PAD_TOKEN))

    print(f"   Source indices: {src_indices[:10]}... (showing first 10)")
    print(f"   Source tensor shape: {src.shape}")
    print(f"   Source mask shape: {src_mask.shape}")

    with torch.no_grad():
        # Test encoder
        encoder_output = model.encoder(src, src_mask)
        print(f"   Encoder output shape: {encoder_output.shape}")
        print(f"   Encoder output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")

        # Test decoder step by step
        tgt_indices = [tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)]
        print(f"\n   Decoder step-by-step:")
        print(f"   SOS token: {tgt_vocab.SOS_TOKEN} (idx: {tgt_indices[0]})")

        for step in range(10):  # First 10 steps
            tgt = torch.tensor([tgt_indices], device=device)
            tgt_mask = create_look_ahead_mask(tgt, tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN))

            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)

            # Analyze output distribution
            next_token_logits = decoder_output[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, 5)

            print(f"     Step {step+1}:")
            print(f"       Current sequence: {[tgt_vocab.get_word(idx) for idx in tgt_indices]}")
            print(f"       Top predictions:")
            for i in range(5):
                word = tgt_vocab.get_word(top_indices[i].item())
                prob = top_probs[i].item()
                print(f"         {word}: {prob:.4f}")

            # Get next token
            next_token = torch.argmax(next_token_logits).item()
            next_word = tgt_vocab.get_word(next_token)
            tgt_indices.append(next_token)

            print(f"       Selected: {next_word} (idx: {next_token})")

            # Check for issues
            if next_word == tgt_vocab.get_word(tgt_indices[-2]):  # Repetition
                print(f"       ⚠️  REPETITION DETECTED!")

            if next_token == tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN):
                print(f"       ✅ EOS reached")
                break

            if step > 5 and all(tgt_vocab.get_word(idx) == "the" for idx in tgt_indices[1:]):
                print(f"       ❌ STUCK GENERATING 'the'!")
                break

        # Final translation
        final_translation = indices_to_sentence(tgt_indices, tgt_vocab)
        print(f"\n   Final translation: '{final_translation}'")

        # Analyze why model is stuck
        print(f"\n🔍 ISSUE ANALYSIS:")

        if final_translation.strip() == "":
            print("   ❌ Empty output - model generates EOS immediately")
            print("   💡 Possible causes:")
            print("      - Model not trained properly")
            print("      - Wrong vocabulary during inference")
            print("      - Decoder initialization issue")

        elif "the the the" in final_translation:
            print("   ❌ Repetitive output - model stuck in loop")
            print("   💡 Possible causes:")
            print("      - Poor training convergence")
            print("      - Label smoothing too aggressive")
            print("      - Attention collapse")
            print("      - Learning rate too high during training")

        elif "<UNK>" in final_translation:
            print("   ⚠️  Many UNK tokens - vocabulary issues")
            print("   💡 Possible causes:")
            print("      - Input words not in vocabulary")
            print("      - Vocabulary mismatch between train/test")

        # Check model weights
        print(f"\n🔍 MODEL WEIGHT ANALYSIS:")

        # Check if weights are reasonable
        total_params = 0
        zero_params = 0
        large_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-8).sum().item()
            large_params += (param.abs() > 10).sum().item()

        print(f"   Total parameters: {total_params:,}")
        print(f"   Near-zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
        print(f"   Large parameters (>10): {large_params:,} ({large_params/total_params*100:.2f}%)")

        if zero_params / total_params > 0.5:
            print("   ❌ Too many zero weights - model might not be trained")

        if large_params / total_params > 0.1:
            print("   ⚠️  Many large weights - possible training instability")

    print(f"\n💡 RECOMMENDATIONS:")
    if len(src_vocab) < 1000:
        print("   1. ❌ CRITICAL: Retrain model with fixed vocabulary (target 10,000 words)")
    else:
        print("   1. ✅ Vocabulary size looks good")

    print("   2. 🔍 Check training logs for convergence issues")
    print("   3. 🎯 Try different decoding strategies (temperature, top-k)")
    print("   4. 📊 Evaluate on more test sentences")
    print("   5. 🔧 Consider adjusting model hyperparameters")

if __name__ == "__main__":
    debug_model_translation()
