import torch
import torch.nn as nn
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

from utils import (load_data, split_data, sentence_to_indices, indices_to_sentence,
                   calculate_bleu, create_padding_mask, create_look_ahead_mask)
from decoder import Transformer, GreedyDecoder, BeamSearchDecoder, TopKDecoder

def parse_args():
    parser = argparse.ArgumentParser(description='Test Transformer for Machine Translation')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./EUbookshop/',
                        help='Directory containing the dataset')
    parser.add_argument('--src_file', type=str, default='EUbookshop.fi',
                        help='Source language file (Finnish)')
    parser.add_argument('--tgt_file', type=str, default='EUbookshop.en',
                        help='Target language file (English)')

    # Decoding arguments
    parser.add_argument('--decoding_strategy', type=str, default='greedy',
                        choices=['greedy', 'beam_search', 'top_k'],
                        help='Decoding strategy to use')
    parser.add_argument('--beam_size', type=int, default=4,
                        help='Beam size for beam search')
    parser.add_argument('--top_k', type=int, default=40,
                        help='k value for top-k sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for top-k sampling')
    parser.add_argument('--max_decode_length', type=int, default=100,
                        help='Maximum length for decoding')

    # Other arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--save_results', type=str, default='./results/',
                        help='Directory to save results')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of translation examples to show')

    return parser.parse_args()

class TransformerTester:
    def __init__(self, model, decoder, src_vocab, tgt_vocab, args):
        self.model = model
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args

    def translate_sentences(self, src_sentences):
        """Translate a list of source sentences"""
        self.model.eval()
        translations = []

        with torch.no_grad():
            for i in tqdm(range(0, len(src_sentences), self.args.batch_size),
                         desc="Translating"):
                batch_sentences = src_sentences[i:i + self.args.batch_size]

                # Convert sentences to indices
                src_batch = []
                for sentence in batch_sentences:
                    src_indices = sentence_to_indices(sentence, self.src_vocab,
                                                    self.args.max_decode_length)
                    src_batch.append(src_indices)

                # Pad sequences to same length
                max_len = max(len(seq) for seq in src_batch)
                padded_batch = []
                for seq in src_batch:
                    padded_seq = seq + [self.src_vocab.get_idx(self.src_vocab.PAD_TOKEN)] * (max_len - len(seq))
                    padded_batch.append(padded_seq)

                src_tensor = torch.tensor(padded_batch).to(self.args.device)

                # Create source mask
                src_mask = self.model.make_src_mask(src_tensor)

                # Decode
                if hasattr(self.decoder, 'decode'):
                    decoded_batch = self.decoder.decode(src_tensor, src_mask)
                else:
                    # For simple greedy decoding
                    decoded_batch = self.greedy_decode(src_tensor, src_mask)

                # Convert back to sentences
                for decoded_seq in decoded_batch:
                    translation = indices_to_sentence(decoded_seq.cpu().numpy(),
                                                    self.tgt_vocab)
                    translations.append(translation)

        return translations

    def greedy_decode(self, src, src_mask):
        """Simple greedy decoding implementation"""
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.model.encoder(src, src_mask)

        # Initialize with SOS token
        decoder_input = torch.full((batch_size, 1),
                                 self.tgt_vocab.get_idx(self.tgt_vocab.SOS_TOKEN),
                                 device=device)

        for _ in range(self.args.max_decode_length):
            tgt_mask = self.model.make_tgt_mask(decoder_input)
            decoder_output = self.model.decoder(decoder_input, encoder_output,
                                              src_mask, tgt_mask)

            # Get next token
            next_token_logits = decoder_output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Check if all sequences have EOS token
            if (next_token.squeeze(-1) == self.tgt_vocab.get_idx(self.tgt_vocab.EOS_TOKEN)).all():
                break

        return decoder_input

    def evaluate_bleu(self, src_sentences, tgt_sentences):
        """Evaluate BLEU score on test set"""
        print("Generating translations for BLEU evaluation...")
        predictions = self.translate_sentences(src_sentences)

        # Clean up predictions and references
        clean_predictions = []
        clean_references = []

        for pred, ref in zip(predictions, tgt_sentences):
            # Remove special tokens and extra whitespace
            pred_clean = pred.strip()
            ref_clean = ref.strip()

            if pred_clean and ref_clean:
                clean_predictions.append(pred_clean)
                clean_references.append(ref_clean)

        # Calculate BLEU score
        bleu_score = calculate_bleu(clean_predictions, clean_references)
        print(f"BLEU Score: {bleu_score:.4f}")

        return bleu_score, clean_predictions, clean_references

    def show_examples(self, src_sentences, tgt_sentences, predictions, num_examples=None):
        """Show translation examples"""
        if num_examples is None:
            num_examples = self.args.num_examples

        print(f"\n{'='*80}")
        print(f"Translation Examples ({self.args.decoding_strategy})")
        print(f"{'='*80}")

        for i in range(min(num_examples, len(src_sentences))):
            print(f"\nExample {i+1}:")
            print(f"Source (Finnish): {src_sentences[i]}")
            print(f"Reference (English): {tgt_sentences[i]}")
            print(f"Translation: {predictions[i]}")
            print("-" * 40)

    def save_results(self, results):
        """Save evaluation results"""
        os.makedirs(self.args.save_results, exist_ok=True)

        filename = f"results_{self.args.decoding_strategy}_{os.path.basename(self.args.model_path)}.json"
        filepath = os.path.join(self.args.save_results, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")

def load_model_and_vocab(model_path, device):
    """Load trained model and vocabularies"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model arguments
    model_args = checkpoint['args']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']

    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        d_ff=model_args.d_ff,
        max_seq_len=model_args.max_seq_len,
        dropout=model_args.dropout,
        pos_encoding_type=model_args.pos_encoding_type
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab, model_args

def create_decoder(decoder_type, model, args, tgt_vocab):
    """Create the specified decoder"""
    sos_idx = tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)
    eos_idx = tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN)
    pad_idx = tgt_vocab.get_idx(tgt_vocab.PAD_TOKEN)

    if decoder_type == 'greedy':
        return GreedyDecoder(model, args.max_decode_length, sos_idx, eos_idx)
    elif decoder_type == 'beam_search':
        return BeamSearchDecoder(model, args.beam_size, args.max_decode_length,
                               sos_idx, eos_idx, pad_idx)
    elif decoder_type == 'top_k':
        return TopKDecoder(model, args.top_k, args.max_decode_length,
                         sos_idx, eos_idx, args.temperature)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

def main():
    args = parse_args()

    # Load model and vocabularies
    model, src_vocab, tgt_vocab, model_args = load_model_and_vocab(args.model_path, args.device)

    # Update args with model max_decode_length if needed
    if hasattr(model_args, 'max_seq_len'):
        args.max_decode_length = min(args.max_decode_length, model_args.max_seq_len)

    print(f"Model loaded successfully!")
    print(f"Positional encoding: {model_args.pos_encoding_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load test data
    print("Loading test data...")
    src_file = os.path.join(args.data_dir, args.src_file)
    tgt_file = os.path.join(args.data_dir, args.tgt_file)

    src_sentences, tgt_sentences = load_data(src_file, tgt_file)

    # Split data to get test set (use same split as training)
    _, _, (test_src, test_tgt) = split_data(src_sentences, tgt_sentences,
                                           train_ratio=0.8, val_ratio=0.1)

    print(f"Test set size: {len(test_src)}")

    # Create decoder
    decoder = create_decoder(args.decoding_strategy, model, args, tgt_vocab)

    # Create tester
    tester = TransformerTester(model, decoder, src_vocab, tgt_vocab, args)

    # Evaluate on test set
    print(f"\nEvaluating with {args.decoding_strategy} decoding...")
    bleu_score, predictions, references = tester.evaluate_bleu(test_src, test_tgt)

    # Show examples
    tester.show_examples(test_src, test_tgt, predictions)

    # Prepare results
    results = {
        'model_path': args.model_path,
        'positional_encoding': model_args.pos_encoding_type,
        'decoding_strategy': args.decoding_strategy,
        'bleu_score': bleu_score,
        'test_set_size': len(test_src),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'decoding_params': {
            'beam_size': args.beam_size if args.decoding_strategy == 'beam_search' else None,
            'top_k': args.top_k if args.decoding_strategy == 'top_k' else None,
            'temperature': args.temperature if args.decoding_strategy == 'top_k' else None,
        },
        'examples': [
            {
                'source': test_src[i],
                'reference': test_tgt[i],
                'translation': predictions[i]
            }
            for i in range(min(args.num_examples, len(test_src)))
        ]
    }

    # Save results
    tester.save_results(results)

    print(f"\nEvaluation completed!")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Decoding Strategy: {args.decoding_strategy}")
    if args.decoding_strategy == 'beam_search':
        print(f"Beam Size: {args.beam_size}")
    elif args.decoding_strategy == 'top_k':
        print(f"Top-k: {args.top_k}, Temperature: {args.temperature}")

if __name__ == '__main__':
    main()
