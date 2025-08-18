import torch
import argparse
from utils import sentence_to_indices, indices_to_sentence
from decoder import Transformer, GreedyDecoder

def parse_args():
    parser = argparse.ArgumentParser(description='Demo script for single translation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True,
                        help='Finnish text to translate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    return parser.parse_args()

def load_model(model_path, device):
    """Load model and vocabularies from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model arguments and vocabularies
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

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab

def translate_text(model, text, src_vocab, tgt_vocab, device, max_length=100):
    """Translate a single text"""
    # Convert text to indices
    src_indices = sentence_to_indices(text, src_vocab, max_length)
    src_tensor = torch.tensor([src_indices]).to(device)

    # Create decoder
    sos_idx = tgt_vocab.get_idx(tgt_vocab.SOS_TOKEN)
    eos_idx = tgt_vocab.get_idx(tgt_vocab.EOS_TOKEN)
    decoder = GreedyDecoder(model, max_length, sos_idx, eos_idx)

    # Translate
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        decoded = decoder.decode(src_tensor, src_mask)

    # Convert back to text
    translation = indices_to_sentence(decoded[0].cpu().numpy(), tgt_vocab)
    return translation

def main():
    args = parse_args()

    print("Loading model...")
    model, src_vocab, tgt_vocab = load_model(args.model_path, args.device)

    print(f"Model loaded successfully!")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    print(f"\nTranslating: '{args.text}'")
    translation = translate_text(model, args.text, src_vocab, tgt_vocab, args.device)

    print(f"Translation: '{translation}'")

if __name__ == '__main__':
    main()
