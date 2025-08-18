#!/bin/bash

# Experiment runner for transformer machine translation
# This script runs training and evaluation for different configurations

echo "Starting Transformer Machine Translation Experiments..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# Set common parameters
DATA_DIR="./EUbookshop/"
BATCH_SIZE=32
NUM_EPOCHS=10
MAX_SEQ_LEN=128

echo "==============================================="
echo "Training Models with Different Positional Encodings"
echo "==============================================="

# Train with RoPE
echo "Training model with RoPE positional encoding..."
python train.py \
    --pos_encoding_type rope \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_len $MAX_SEQ_LEN \
    --save_dir checkpoints \
    --data_dir $DATA_DIR \
    > logs/training_rope.log 2>&1

# Train with Relative Position Bias
echo "Training model with Relative Position Bias..."
python train.py \
    --pos_encoding_type relative_bias \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_len $MAX_SEQ_LEN \
    --save_dir checkpoints \
    --data_dir $DATA_DIR \
    > logs/training_relative_bias.log 2>&1

echo "==============================================="
echo "Testing Models with Different Decoding Strategies"
echo "==============================================="

# Test RoPE model with different decoding strategies
echo "Testing RoPE model..."

# Greedy decoding
echo "  - Greedy decoding..."
python test.py \
    --model_path checkpoints/best_model_rope.pt \
    --decoding_strategy greedy \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rope_greedy.log 2>&1

# Beam search
echo "  - Beam search (beam_size=4)..."
python test.py \
    --model_path checkpoints/best_model_rope.pt \
    --decoding_strategy beam_search \
    --beam_size 4 \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rope_beam.log 2>&1

# Top-k sampling
echo "  - Top-k sampling (k=40)..."
python test.py \
    --model_path checkpoints/best_model_rope.pt \
    --decoding_strategy top_k \
    --top_k 40 \
    --temperature 1.0 \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rope_topk.log 2>&1

# Test Relative Bias model with different decoding strategies
echo "Testing Relative Bias model..."

# Greedy decoding
echo "  - Greedy decoding..."
python test.py \
    --model_path checkpoints/best_model_relative_bias.pt \
    --decoding_strategy greedy \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rel_bias_greedy.log 2>&1

# Beam search
echo "  - Beam search (beam_size=4)..."
python test.py \
    --model_path checkpoints/best_model_relative_bias.pt \
    --decoding_strategy beam_search \
    --beam_size 4 \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rel_bias_beam.log 2>&1

# Top-k sampling
echo "  - Top-k sampling (k=40)..."
python test.py \
    --model_path checkpoints/best_model_relative_bias.pt \
    --decoding_strategy top_k \
    --top_k 40 \
    --temperature 1.0 \
    --data_dir $DATA_DIR \
    --save_results results \
    --batch_size $BATCH_SIZE \
    > logs/test_rel_bias_topk.log 2>&1

echo "==============================================="
echo "Experiments Completed!"
echo "==============================================="

echo "Results saved in:"
echo "  - Checkpoints: checkpoints/"
echo "  - Results: results/"
echo "  - Logs: logs/"
echo ""
echo "To view training curves, check the PNG files in checkpoints/"
echo "To see BLEU scores, check the JSON files in results/"
