# Transformer Machine Translation

This project implements a Transformer model from scratch for Finnish-to-English machine translation.

## Features

- Complete Transformer architecture implemented from scratch
- Two positional encoding strategies:
  - Rotary Positional Embeddings (RoPE)
  - Relative Position Bias
- Three decoding strategies:
  - Greedy Decoding
  - Beam Search
  - Top-k Sampling
- Full training pipeline with teacher forcing
- Comprehensive evaluation with BLEU scores

## Requirements

```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn tqdm
pip install sacrebleu
pip install numpy
```

## Dataset

The dataset consists of parallel Finnish-English sentences from EUbookshop corpus:
- `EUbookshop.fi`: Finnish sentences
- `EUbookshop.en`: English sentences

## Usage

### Training

Train with RoPE positional encoding:
```bash
python train.py --pos_encoding_type rope --num_epochs 20 --batch_size 32
```

Train with Relative Position Bias:
```bash
python train.py --pos_encoding_type relative_bias --num_epochs 20 --batch_size 32
```

### Testing

Test with greedy decoding:
```bash
python test.py --model_path checkpoints/best_model_rope.pt --decoding_strategy greedy
```

Test with beam search:
```bash
python test.py --model_path checkpoints/best_model_rope.pt --decoding_strategy beam_search --beam_size 4
```

Test with top-k sampling:
```bash
python test.py --model_path checkpoints/best_model_rope.pt --decoding_strategy top_k --top_k 40 --temperature 1.0
```

## Training Arguments

- `--pos_encoding_type`: Choose between 'rope', 'relative_bias', or 'sinusoidal'
- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 6)
- `--num_decoder_layers`: Number of decoder layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--num_epochs`: Number of epochs (default: 20)

## Model Architecture

### Encoder
- Multi-head self-attention with configurable positional encoding
- Position-wise feed-forward networks
- Layer normalization and residual connections

### Decoder
- Masked multi-head self-attention
- Multi-head cross-attention with encoder output
- Position-wise feed-forward networks
- Layer normalization and residual connections

### Positional Encodings

1. **RoPE (Rotary Positional Embeddings)**:
   - Applied directly to query and key vectors in attention
   - No additional positional embeddings added to input

2. **Relative Position Bias**:
   - Learnable bias terms added to attention scores
   - Based on relative positions between tokens

## Files Structure

```
anlp_ass1/
├── encoder.py          # Encoder implementation
├── decoder.py          # Decoder implementation
├── train.py           # Training script
├── test.py            # Testing script
├── utils.py           # Utility functions
├── README.md          # This file
├── report.pdf         # Analysis report
├── EUbookshop/        # Dataset directory
│   ├── EUbookshop.fi  # Finnish sentences
│   └── EUbookshop.en  # English sentences
├── checkpoints/       # Model checkpoints
└── results/           # Evaluation results
```

## Pre-trained Models

Download pre-trained models from: [Link will be provided after training]

## Evaluation

The models are evaluated using:
- BLEU scores on test set
- Translation quality comparison across decoding strategies
- Convergence speed analysis between positional encoding methods

## Results

Results will be saved in the `results/` directory and include:
- BLEU scores for each configuration
- Training curves
- Translation examples
- Detailed analysis in `report.pdf`

## Implementation Details

- No pre-built Transformer modules from PyTorch are used
- All components implemented from scratch including:
  - Multi-head attention
  - Positional encodings
  - Encoder and decoder stacks
  - Beam search and top-k sampling
- Label smoothing for better training
- Gradient clipping for stable training
- Learning rate scheduling following the original paper

## Citation

If you use this implementation, please cite:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
