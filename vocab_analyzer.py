#!/usr/bin/env python3
"""
Vocabulary Diagnostic Script
Analyzes the vocabulary in your trained model
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_model_vocabulary(model_path):
    """Analyze vocabulary in a trained model"""

    print(f"🔍 Analyzing vocabulary in: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully!")

        # Extract vocabularies
        src_vocab = checkpoint['src_vocab']  # Finnish
        tgt_vocab = checkpoint['tgt_vocab']  # English

        print("\n" + "="*60)
        print("📊 VOCABULARY ANALYSIS")
        print("="*60)

        # Source vocabulary (Finnish)
        print(f"\n🇫🇮 Finnish Vocabulary:")
        print(f"   Size: {len(src_vocab)} words")
        print(f"   Words:")
        for i, (word, idx) in enumerate(src_vocab.word2idx.items()):
            if i < 20:  # Show first 20 words
                count = src_vocab.word_count.get(word, 0)
                print(f"     {idx:2d}: '{word}' (count: {count})")
            elif i == 20:
                print(f"     ... and {len(src_vocab) - 20} more words")
                break

        # Target vocabulary (English)
        print(f"\n🇬🇧 English Vocabulary:")
        print(f"   Size: {len(tgt_vocab)} words")
        print(f"   Words:")
        for i, (word, idx) in enumerate(tgt_vocab.word2idx.items()):
            if i < 20:  # Show first 20 words
                count = tgt_vocab.word_count.get(word, 0)
                print(f"     {idx:2d}: '{word}' (count: {count})")
            elif i == 20:
                print(f"     ... and {len(tgt_vocab) - 20} more words")
                break

        # Special tokens
        print(f"\n🔧 Special Tokens:")
        print(f"   PAD: '{src_vocab.PAD_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.PAD_TOKEN)})")
        print(f"   SOS: '{src_vocab.SOS_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.SOS_TOKEN)})")
        print(f"   EOS: '{src_vocab.EOS_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.EOS_TOKEN)})")
        print(f"   UNK: '{src_vocab.UNK_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.UNK_TOKEN)})")

        # Analysis
        print(f"\n📈 Analysis:")
        if len(src_vocab) < 100:
            print("   ❌ CRITICAL: Vocabulary too small! This explains the <UNK> tokens.")
            print("   💡 Solution: Retrain with fixed create_vocabulary function")
        elif len(src_vocab) < 1000:
            print("   ⚠️  WARNING: Vocabulary quite small, may have limited translation ability")
        else:
            print("   ✅ Vocabulary size looks reasonable")

        # Test sentence analysis
        print(f"\n🧪 Test Sentence Analysis:")
        test_sentence = "Ohjelmakaudella 2007-20013 Suomen ESR-ohjelma tukee elinkeinoelämän kilpailukykyä ja työllisyyttä"
        words = test_sentence.lower().split()

        print(f"   Input: '{test_sentence}'")
        print(f"   Word-by-word analysis:")
        for word in words:
            if word in src_vocab.word2idx:
                idx = src_vocab.get_idx(word)
                print(f"     '{word}' → Found (idx: {idx})")
            else:
                print(f"     '{word}' → <UNK> (not in vocabulary)")

        unknown_count = sum(1 for word in words if word not in src_vocab.word2idx)
        print(f"   Result: {unknown_count}/{len(words)} words are unknown ({unknown_count/len(words)*100:.1f}%)")

        if unknown_count == len(words):
            print("   ❌ ALL words are unknown - this is why you get only <UNK> tokens!")

    except Exception as e:
        print(f"❌ Error analyzing model: {e}")

def find_model_file():
    """Find model file"""
    possible_paths = [
        './best_model.pt',
        './final_model.pt',
        './model.pt',
        os.path.expanduser('~/Downloads/best_model.pt'),
        os.path.expanduser('~/Desktop/best_model.pt')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def main():
    """Main function"""
    print("🔍 Model Vocabulary Analyzer")
    print("="*60)

    # Find model
    model_path = find_model_file()

    if model_path is None:
        print("❌ No model file found automatically.")
        model_path = input("Enter path to your .pt model file: ").strip()

        if not model_path or not os.path.exists(model_path):
            print("❌ Invalid path.")
            return
    else:
        print(f"✅ Found model: {model_path}")

    # Analyze
    analyze_model_vocabulary(model_path)

    print("\n" + "="*60)
    print("💡 SOLUTIONS:")
    print("="*60)
    print("1. 🔧 Fix the create_vocabulary function (already done in utils.py)")
    print("2. 🏋️  Retrain your model with the fixed vocabulary function")
    print("3. 📊 You should see vocab sizes of ~10,000 words after retraining")
    print("4. ✅ Then translations will work much better!")

if __name__ == "__main__":
    main()
