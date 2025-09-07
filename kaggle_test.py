#!/usr/bin/env python3
"""
Transformer Testing Script for Kaggle
This script loads a trained model and evaluates it on test data
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Kaggle paths - matches your input structure
KAGGLE_INPUT_BASE = "/kaggle/input"
DATA_PATH = f"{KAGGLE_INPUT_BASE}/anlp-ass1"  # Your dataset name

# Add paths for imports
sys.path.append(DATA_PATH)
sys.path.append("/kaggle/working")

print(f"📁 Data path: {DATA_PATH}")
print(f"📁 Working directory: /kaggle/working")

# Import your modules
try:
    from encoder import TransformerEncoder
    from decoder import TransformerDecoder, Transformer
    from utils import (
        load_data, split_data, create_vocabulary, TransformerDataset,
        create_padding_mask, create_look_ahead_mask, calculate_bleu,
        indices_to_sentence, Vocabulary
    )
    print("✅ Successfully imported custom modules")
    modules_imported = True
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure your encoder.py, decoder.py, and utils.py are in the Kaggle input")
    print("Available files in DATA_PATH:")
    try:
        files = os.listdir(DATA_PATH)
        for f in sorted(files):
            print(f"  {f}")
    except:
        print("  Could not list files")
    modules_imported = False

# Add safe globals for PyTorch loading
try:
    torch.serialization.add_safe_globals([Vocabulary])
except Exception:
    pass

class ModelTester:
    def __init__(self, model_path, device='cuda'):
        """Initialize the model tester for Kaggle"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # Load checkpoint
        print(f"📥 Loading model from {model_path}")

        # Handle PyTorch's weights_only security feature
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading with weights_only=False: {e}")
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

        print(f"✅ Model loaded successfully!")
        print(f"🔧 Positional encoding: {self.config['pos_encoding_type']}")
        print(f"📊 Source vocab size: {len(self.src_vocab):,}")
        print(f"📊 Target vocab size: {len(self.tgt_vocab):,}")
        print(f"🤖 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def sentence_to_indices(self, sentence, vocab, max_length=None):
        """Convert sentence to indices"""
        words = sentence.lower().strip().split()
        indices = [vocab.get_idx(word) for word in words]

        if max_length is None:
            max_length = self.config['max_seq_len']

        if len(indices) > max_length - 2:
            indices = indices[:max_length - 2]

        # Add SOS and EOS
        indices = [vocab.get_idx(vocab.SOS_TOKEN)] + indices + [vocab.get_idx(vocab.EOS_TOKEN)]

        # Pad if necessary
        while len(indices) < max_length:
            indices.append(vocab.get_idx(vocab.PAD_TOKEN))

        return indices

    def translate_greedy(self, sentence, max_length=50):
        """Translate using greedy decoding"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)
            tgt_indices = [self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN)]

            for _ in range(max_length):
                tgt = torch.tensor([tgt_indices], device=self.device)
                tgt_mask = create_look_ahead_mask(tgt, self.tgt_vocab.get_idx(self.tgt_vocab.PAD_TOKEN))

                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                next_token = torch.argmax(decoder_output[0, -1, :]).item()
                tgt_indices.append(next_token)

                if next_token == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN):
                    break

        return indices_to_sentence(tgt_indices, self.tgt_vocab)

    def translate_beam_search(self, sentence, beam_size=4, max_length=50):
        """Translate using beam search"""
        self.model.eval()

        src_indices = self.sentence_to_indices(sentence, self.src_vocab)
        src = torch.tensor([src_indices], device=self.device)
        src_mask = create_padding_mask(src, self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN))

        with torch.no_grad():
            encoder_output = self.model.encoder(src, src_mask)
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

                    decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                    log_probs = F.log_softmax(decoder_output[0, -1, :], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)

                    for j in range(beam_size):
                        new_beam = beam + [top_k_indices[j].item()]
                        new_score = beam_scores[i] + top_k_probs[j].item()
                        candidates.append((new_score, new_beam))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = [cand[1] for cand in candidates[:beam_size]]
                beam_scores = [cand[0] for cand in candidates[:beam_size]]

                if all(beam[-1] == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN) for beam in beams):
                    break

        return indices_to_sentence(beams[0], self.tgt_vocab)

    def calculate_accuracy_metrics(self, predictions, references):
        """Calculate accuracy metrics"""
        exact_matches = 0
        word_level_correct = 0
        total_words = 0

        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

            pred_words = pred.strip().lower().split()
            ref_words = ref.strip().lower().split()

            for i in range(min(len(pred_words), len(ref_words))):
                if pred_words[i] == ref_words[i]:
                    word_level_correct += 1
            total_words += len(ref_words)

        exact_match_acc = (exact_matches / len(predictions)) * 100
        word_level_acc = (word_level_correct / total_words) * 100 if total_words > 0 else 0

        return exact_match_acc, word_level_acc

    def test_sample_sentences(self):
        """Test on sample sentences"""
        sample_sentences = [
            "Hyvää huomenta",        # Good morning
            "Kiitos paljon",         # Thank you very much
            "Nähdään myöhemmin",     # See you later
            "Mikä on nimesi?",       # What is your name?
            "Pidän kahvista",        # I like coffee
            "Sää on kaunis tänään",  # The weather is beautiful today
            "Voitko auttaa minua?",  # Can you help me?
            "Olen opiskelija",       # I am a student
        ]

        print("\n" + "="*60)
        print("🧪 SAMPLE SENTENCE TESTING")
        print("="*60)

        results = []
        for i, sentence in enumerate(sample_sentences):
            print(f"\n{i+1}. Finnish: {sentence}")

            try:
                greedy_trans = self.translate_greedy(sentence)
                beam_trans = self.translate_beam_search(sentence, beam_size=4)

                print(f"   Greedy: {greedy_trans}")
                print(f"   Beam:   {beam_trans}")

                results.append({
                    'Finnish': sentence,
                    'Greedy': greedy_trans,
                    'Beam': beam_trans
                })

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'Finnish': sentence,
                    'Greedy': f"Error: {e}",
                    'Beam': f"Error: {e}"
                })

        # Save results
        df = pd.DataFrame(results)
        df.to_csv('/kaggle/working/sample_translations.csv', index=False)
        print(f"\n💾 Sample translations saved to /kaggle/working/sample_translations.csv")

        return results

    def comprehensive_evaluation(self, use_full_test=True, greedy_only=False):
        """Comprehensive evaluation on test set"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE MODEL EVALUATION")
        print("="*80)

        # Load test data
        print("📁 Loading test data...")
        try:
            en_file = f"{DATA_PATH}/EUbookshop.en"
            fi_file = f"{DATA_PATH}/EUbookshop.fi"

            if not os.path.exists(en_file) or not os.path.exists(fi_file):
                print(f"❌ Data files not found:")
                print(f"   Expected: {en_file}")
                print(f"   Expected: {fi_file}")
                return None

            en_sentences, fi_sentences = load_data(en_file, fi_file)
            train_data, val_data, test_data = split_data(en_sentences, fi_sentences, train_ratio=0.8, val_ratio=0.1)

            if use_full_test:
                test_src = test_data[1]  # Finnish
                test_tgt = test_data[0]  # English
                print(f"✅ Using full test set: {len(test_src):,} sentences")
            else:
                test_src = test_data[1][:100]
                test_tgt = test_data[0][:100]
                print(f"✅ Using test subset: {len(test_src)} sentences")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        predictions_greedy = []
        predictions_beam = []
        references = []

        print(f"\n📊 Test set size: {len(test_src)} sentence pairs")
        print(f"🤖 Model: {self.config['pos_encoding_type']} positional encoding")
        print(f"⚙️  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if greedy_only:
            print("🚀 Mode: FAST (Greedy decoding only)")
        else:
            print("🔍 Mode: FULL (Greedy + Beam search)")
        print("="*80)

        if greedy_only:
            # Fast mode: Only greedy
            print("⚡ Generating translations with greedy decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Greedy")):
                pred_greedy = self.translate_greedy(src_sentence)
                predictions_greedy.append(pred_greedy)
                references.append(test_tgt[i])

                if i < 10:  # Show first 10 examples
                    print(f"\nExample {i+1}:")
                    print(f"  Source (FI): {src_sentence}")
                    print(f"  Reference:   {test_tgt[i]}")
                    print(f"  Greedy:      {pred_greedy}")

            predictions_beam = predictions_greedy.copy()

        else:
            # Full mode: Both algorithms
            print("🔍 Phase 1/2: Greedy decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Greedy")):
                pred_greedy = self.translate_greedy(src_sentence)
                predictions_greedy.append(pred_greedy)
                references.append(test_tgt[i])

            print("🔍 Phase 2/2: Beam search decoding...")
            for i, src_sentence in enumerate(tqdm(test_src, desc="Beam")):
                pred_beam = self.translate_beam_search(src_sentence, beam_size=4)
                predictions_beam.append(pred_beam)

                if i < 10:  # Show first 10 examples
                    print(f"\nExample {i+1}:")
                    print(f"  Source (FI): {src_sentence}")
                    print(f"  Reference:   {test_tgt[i]}")
                    print(f"  Greedy:      {predictions_greedy[i]}")
                    print(f"  Beam:        {pred_beam}")

        # Calculate metrics
        print("\n" + "="*80)
        print("📈 EVALUATION RESULTS")
        print("="*80)

        # BLEU scores
        try:
            bleu_greedy = calculate_bleu(predictions_greedy, references)
            bleu_beam = calculate_bleu(predictions_beam, references)

            print(f"\n📊 BLEU SCORES:")
            print(f"  Greedy Decoding: {bleu_greedy:.4f}")
            print(f"  Beam Search:     {bleu_beam:.4f}")
            print(f"  Improvement:     {bleu_beam - bleu_greedy:+.4f}")

        except Exception as e:
            print(f"❌ Error calculating BLEU score: {e}")
            bleu_greedy = bleu_beam = 0

        # Accuracy metrics
        exact_match_greedy, word_acc_greedy = self.calculate_accuracy_metrics(predictions_greedy, references)
        exact_match_beam, word_acc_beam = self.calculate_accuracy_metrics(predictions_beam, references)

        print(f"\n🎯 ACCURACY METRICS:")
        print(f"  Exact Match Accuracy:")
        print(f"    Greedy:  {exact_match_greedy:.2f}%")
        print(f"    Beam:    {exact_match_beam:.2f}%")
        print(f"  Word-level Accuracy:")
        print(f"    Greedy:  {word_acc_greedy:.2f}%")
        print(f"    Beam:    {word_acc_beam:.2f}%")

        # Sentence length analysis
        avg_src_len = sum(len(s.split()) for s in test_src) / len(test_src)
        avg_ref_len = sum(len(s.split()) for s in references) / len(references)
        avg_pred_greedy_len = sum(len(s.split()) for s in predictions_greedy) / len(predictions_greedy)
        avg_pred_beam_len = sum(len(s.split()) for s in predictions_beam) / len(predictions_beam)

        print(f"\n📏 AVERAGE SENTENCE LENGTHS:")
        print(f"  Source (Finnish):    {avg_src_len:.1f} words")
        print(f"  Reference (English): {avg_ref_len:.1f} words")
        print(f"  Greedy Predictions:  {avg_pred_greedy_len:.1f} words")
        print(f"  Beam Predictions:    {avg_pred_beam_len:.1f} words")

        # Final summary
        print(f"\n🏆 MODEL PERFORMANCE SUMMARY:")
        print(f"  Dataset: Finnish → English")
        print(f"  Test sentences: {len(test_src):,}")
        print(f"  Best BLEU: {max(bleu_greedy, bleu_beam):.4f} ({'Beam' if bleu_beam > bleu_greedy else 'Greedy'})")
        print(f"  Best Accuracy: {max(exact_match_greedy, exact_match_beam):.2f}% ({'Beam' if exact_match_beam > exact_match_greedy else 'Greedy'})")
        print(f"  Vocab sizes: FI={len(self.src_vocab):,}, EN={len(self.tgt_vocab):,}")
        print("="*80)

        # Save detailed results
        results_df = pd.DataFrame({
            'Source_Finnish': test_src,
            'Reference_English': references,
            'Greedy_Translation': predictions_greedy,
            'Beam_Translation': predictions_beam
        })
        results_df.to_csv('/kaggle/working/detailed_test_results.csv', index=False)

        # Save summary metrics
        summary = {
            'test_sentences': len(test_src),
            'bleu_greedy': bleu_greedy,
            'bleu_beam': bleu_beam,
            'exact_match_greedy': exact_match_greedy,
            'exact_match_beam': exact_match_beam,
            'word_accuracy_greedy': word_acc_greedy,
            'word_accuracy_beam': word_acc_beam,
            'avg_source_length': avg_src_len,
            'avg_reference_length': avg_ref_len,
            'avg_greedy_length': avg_pred_greedy_len,
            'avg_beam_length': avg_pred_beam_len,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'positional_encoding': self.config['pos_encoding_type']
        }

        with open('/kaggle/working/evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"  📄 Detailed results: /kaggle/working/detailed_test_results.csv")
        print(f"  📊 Summary metrics: /kaggle/working/evaluation_summary.json")

        return summary

def main():
    """Main testing function"""

    # Check if modules were imported successfully
    if not modules_imported:
        print("❌ Cannot proceed without required modules")
        print("📋 To fix this:")
        print("1. Make sure encoder.py, decoder.py, and utils.py are uploaded to your Kaggle dataset")
        print("2. Check that the files are in the correct path")
        print("3. Ensure sacrebleu is installed (script will try to install it)")
        return
    print("🚀 Starting Transformer Testing on Kaggle")
    print("="*60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU not available - using CPU")
        print("   💡 In Kaggle, make sure to enable GPU in Settings > Accelerator")

    # Model paths to try - modify this to test specific checkpoints
    model_paths = [
        '/kaggle/working/best_model.pt',      # Best validation performance (recommended)
        '/kaggle/working/final_model.pt',     # Final training state
        '/kaggle/working/checkpoint_epoch_2.pt',  # Specific epoch
        f'{DATA_PATH}/best_model.pt',         # If model is in dataset
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ No model found! Please either:")
        print("   1. Run training first in the same session, or")
        print("   2. Upload your trained model to Kaggle input")
        return

    print(f"📥 Using model: {model_path}")

    try:
        # Initialize tester
        tester = ModelTester(model_path, device)

        # Test sample sentences
        print("\n" + "="*60)
        print("🧪 Testing sample sentences...")
        sample_results = tester.test_sample_sentences()

        # Comprehensive evaluation
        print("\n" + "="*60)
        print("🔬 Running comprehensive evaluation...")

        # You can modify these parameters:
        USE_FULL_TEST = True    # Set to False for faster testing with subset
        GREEDY_ONLY = False     # Set to True for faster greedy-only testing

        summary = tester.comprehensive_evaluation(
            use_full_test=USE_FULL_TEST,
            greedy_only=GREEDY_ONLY
        )

        if summary:
            print("\n✅ Testing completed successfully!")
            print(f"🎯 Best BLEU Score: {max(summary['bleu_greedy'], summary['bleu_beam']):.4f}")
            print(f"📊 Test Sentences: {summary['test_sentences']:,}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Important: Update these paths for your Kaggle datasets
    print("🔧 Configuration:")
    print(f"   DATA_PATH: {DATA_PATH}")
    print("="*60)

    main()
