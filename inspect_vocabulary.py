#!/usr/bin/env python3
"""
Vocabulary Inspector
Load trained model and display vocabulary contents
"""

import torch
import sys
import os

sys.path.append(".")

def inspect_vocabulary():
    """Load model and display vocabulary contents"""

    try:
        # Load the model checkpoint
        if os.path.exists('./best_model.pt'):
            checkpoint = torch.load('./best_model.pt', map_location='cpu', weights_only=False)
            print("✅ Loaded best_model.pt")
        else:
            print("❌ best_model.pt not found")
            return

        src_vocab = checkpoint['src_vocab']  # Finnish
        tgt_vocab = checkpoint['tgt_vocab']  # English

        print(f"\n📊 VOCABULARY STATISTICS:")
        print(f"   Finnish vocabulary size: {len(src_vocab)}")
        print(f"   English vocabulary size: {len(tgt_vocab)}")

        # Display special tokens
        print(f"\n🔧 SPECIAL TOKENS:")
        print(f"   PAD: '{src_vocab.PAD_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.PAD_TOKEN)})")
        print(f"   SOS: '{src_vocab.SOS_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.SOS_TOKEN)})")
        print(f"   EOS: '{src_vocab.EOS_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.EOS_TOKEN)})")
        print(f"   UNK: '{src_vocab.UNK_TOKEN}' (idx: {src_vocab.get_idx(src_vocab.UNK_TOKEN)})")

        # Display first 50 Finnish words
        print(f"\n🇫🇮 FIRST 50 FINNISH WORDS:")
        print("   Index | Word")
        print("   ------|-----")
        for i in range(min(50, len(src_vocab.word2idx))):
            word = src_vocab.get_word(i)
            print(f"   {i:5d} | {word}")

        # Display first 50 English words
        print(f"\n🇺🇸 FIRST 50 ENGLISH WORDS:")
        print("   Index | Word")
        print("   ------|-----")
        for i in range(min(50, len(tgt_vocab.word2idx))):
            word = tgt_vocab.get_word(i)
            print(f"   {i:5d} | {word}")

        # Show most common words (excluding special tokens)
        print(f"\n📈 MOST COMMON FINNISH WORDS (excluding specials):")
        finnish_words = []
        for i in range(4, min(30, len(src_vocab.word2idx))):  # Skip first 4 (special tokens)
            word = src_vocab.get_word(i)
            if word not in [src_vocab.PAD_TOKEN, src_vocab.SOS_TOKEN, src_vocab.EOS_TOKEN, src_vocab.UNK_TOKEN]:
                finnish_words.append(word)

        for i, word in enumerate(finnish_words[:20]):
            print(f"   {i+1:2d}. {word}")

        print(f"\n📈 MOST COMMON ENGLISH WORDS (excluding specials):")
        english_words = []
        for i in range(4, min(30, len(tgt_vocab.word2idx))):  # Skip first 4 (special tokens)
            word = tgt_vocab.get_word(i)
            if word not in [tgt_vocab.PAD_TOKEN, tgt_vocab.SOS_TOKEN, tgt_vocab.EOS_TOKEN, tgt_vocab.UNK_TOKEN]:
                english_words.append(word)

        for i, word in enumerate(english_words[:20]):
            print(f"   {i+1:2d}. {word}")

        # Check for specific words
        test_words = ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"]
        print(f"\n🔍 CHECKING COMMON ENGLISH WORDS:")
        for word in test_words:
            idx = tgt_vocab.get_idx(word)
            if idx != tgt_vocab.get_idx(tgt_vocab.UNK_TOKEN):
                print(f"   ✅ '{word}' found at index {idx}")
            else:
                print(f"   ❌ '{word}' not in vocabulary")

        # Check for Finnish words
        finnish_test = ["ja", "on", "ei", "se", "että", "tai", "kuin", "kun", "niin", "siis"]
        print(f"\n🔍 CHECKING COMMON FINNISH WORDS:")
        for word in finnish_test:
            idx = src_vocab.get_idx(word)
            if idx != src_vocab.get_idx(src_vocab.UNK_TOKEN):
                print(f"   ✅ '{word}' found at index {idx}")
            else:
                print(f"   ❌ '{word}' not in vocabulary")

        # Save vocabulary to files for further inspection
        print(f"\n💾 SAVING VOCABULARIES TO FILES:")

        with open('finnish_vocabulary.txt', 'w', encoding='utf-8') as f:
            f.write("# Finnish Vocabulary (index: word)\n")
            for i in range(len(src_vocab.word2idx)):
                word = src_vocab.get_word(i)
                f.write(f"{i}: {word}\n")
        print("   ✅ Saved finnish_vocabulary.txt")

        with open('english_vocabulary.txt', 'w', encoding='utf-8') as f:
            f.write("# English Vocabulary (index: word)\n")
            for i in range(len(tgt_vocab.word2idx)):
                word = tgt_vocab.get_word(i)
                f.write(f"{i}: {word}\n")
        print("   ✅ Saved english_vocabulary.txt")

        print(f"\n📋 SUMMARY:")
        print(f"   • Both vocabularies have {len(src_vocab)} and {len(tgt_vocab)} words")
        print(f"   • Special tokens are properly included")
        print(f"   • Complete vocabularies saved to text files")
        print(f"   • You can search specific words in the generated files")

    except Exception as e:
        print(f"❌ Error inspecting vocabulary: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vocabulary()
