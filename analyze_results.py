import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_info(filepath):
    """Load training information from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_results(filepath):
    """Load test results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_curves_comparison():
    """Plot training curves for both positional encoding methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Load training information
    try:
        rope_info = load_training_info('checkpoints/training_info_rope.json')
        rel_bias_info = load_training_info('checkpoints/training_info_relative_bias.json')

        # Plot training losses
        ax1.plot(rope_info['train_losses'], label='RoPE - Training', linewidth=2)
        ax1.plot(rope_info['val_losses'], label='RoPE - Validation', linewidth=2)
        ax1.plot(rel_bias_info['train_losses'], label='Rel. Bias - Training', linewidth=2, linestyle='--')
        ax1.plot(rel_bias_info['val_losses'], label='Rel. Bias - Validation', linewidth=2, linestyle='--')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation losses only (for clarity)
        ax2.plot(rope_info['val_losses'], label='RoPE', linewidth=3, marker='o', markersize=4)
        ax2.plot(rel_bias_info['val_losses'], label='Relative Bias', linewidth=3, marker='s', markersize=4)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print convergence analysis
        print("Convergence Analysis:")
        print(f"RoPE - Best Validation Loss: {rope_info['best_val_loss']:.4f}")
        print(f"Relative Bias - Best Validation Loss: {rel_bias_info['best_val_loss']:.4f}")

        # Find epoch of best performance
        rope_best_epoch = np.argmin(rope_info['val_losses']) + 1
        rel_bias_best_epoch = np.argmin(rel_bias_info['val_losses']) + 1

        print(f"RoPE - Best epoch: {rope_best_epoch}")
        print(f"Relative Bias - Best epoch: {rel_bias_best_epoch}")

        if rope_best_epoch < rel_bias_best_epoch:
            print("RoPE converged faster!")
        elif rel_bias_best_epoch < rope_best_epoch:
            print("Relative Bias converged faster!")
        else:
            print("Both methods converged at the same rate.")

    except FileNotFoundError as e:
        print(f"Training info file not found: {e}")

def create_bleu_scores_table():
    """Create a table of BLEU scores for different configurations"""
    results_dir = Path('results')

    # Find all result files
    result_files = list(results_dir.glob('results_*.json'))

    if not result_files:
        print("No result files found. Run experiments first.")
        return

    # Collect results
    results_data = []

    for file in result_files:
        try:
            result = load_results(file)
            results_data.append({
                'file': file.name,
                'pos_encoding': result.get('positional_encoding', 'unknown'),
                'decoding': result.get('decoding_strategy', 'unknown'),
                'bleu_score': result.get('bleu_score', 0.0),
                'beam_size': result.get('decoding_params', {}).get('beam_size'),
                'top_k': result.get('decoding_params', {}).get('top_k'),
                'temperature': result.get('decoding_params', {}).get('temperature')
            })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not results_data:
        print("No valid result files found.")
        return

    # Sort results
    results_data.sort(key=lambda x: (x['pos_encoding'], x['decoding']))

    # Print table
    print("\n" + "="*80)
    print("BLEU SCORES TABLE")
    print("="*80)
    print(f"{'Pos. Encoding':<15} {'Decoding':<12} {'BLEU Score':<12} {'Parameters':<20}")
    print("-"*80)

    for result in results_data:
        pos_enc = result['pos_encoding']
        decoding = result['decoding']
        bleu = result['bleu_score']

        # Format parameters
        params = ""
        if result['beam_size'] is not None:
            params = f"beam_size={result['beam_size']}"
        elif result['top_k'] is not None:
            params = f"k={result['top_k']}, T={result['temperature']}"

        print(f"{pos_enc:<15} {decoding:<12} {bleu:<12.4f} {params:<20}")

    print("="*80)

    # Find best configurations
    best_overall = max(results_data, key=lambda x: x['bleu_score'])
    print(f"\nBest Overall: {best_overall['pos_encoding']} + {best_overall['decoding']} "
          f"(BLEU: {best_overall['bleu_score']:.4f})")

    # Best per positional encoding
    rope_results = [r for r in results_data if r['pos_encoding'] == 'rope']
    rel_bias_results = [r for r in results_data if r['pos_encoding'] == 'relative_bias']

    if rope_results:
        best_rope = max(rope_results, key=lambda x: x['bleu_score'])
        print(f"Best RoPE: {best_rope['decoding']} (BLEU: {best_rope['bleu_score']:.4f})")

    if rel_bias_results:
        best_rel_bias = max(rel_bias_results, key=lambda x: x['bleu_score'])
        print(f"Best Rel. Bias: {best_rel_bias['decoding']} (BLEU: {best_rel_bias['bleu_score']:.4f})")

    # Best per decoding strategy
    decoding_strategies = set(r['decoding'] for r in results_data)
    for strategy in decoding_strategies:
        strategy_results = [r for r in results_data if r['decoding'] == strategy]
        best_strategy = max(strategy_results, key=lambda x: x['bleu_score'])
        print(f"Best {strategy}: {best_strategy['pos_encoding']} (BLEU: {best_strategy['bleu_score']:.4f})")

def plot_bleu_comparison():
    """Create bar plots comparing BLEU scores"""
    results_dir = Path('results')
    result_files = list(results_dir.glob('results_*.json'))

    if not result_files:
        print("No result files found for plotting.")
        return

    # Collect results
    results_data = []
    for file in result_files:
        try:
            result = load_results(file)
            results_data.append({
                'pos_encoding': result.get('positional_encoding', 'unknown'),
                'decoding': result.get('decoding_strategy', 'unknown'),
                'bleu_score': result.get('bleu_score', 0.0)
            })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not results_data:
        return

    # Organize data for plotting
    pos_encodings = sorted(set(r['pos_encoding'] for r in results_data))
    decoding_strategies = sorted(set(r['decoding'] for r in results_data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: BLEU scores by positional encoding
    pos_encoding_scores = {}
    for pos_enc in pos_encodings:
        scores = [r['bleu_score'] for r in results_data if r['pos_encoding'] == pos_enc]
        pos_encoding_scores[pos_enc] = np.mean(scores) if scores else 0

    ax1.bar(pos_encoding_scores.keys(), pos_encoding_scores.values(),
            color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Average BLEU Score')
    ax1.set_title('BLEU Scores by Positional Encoding')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (k, v) in enumerate(pos_encoding_scores.items()):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

    # Plot 2: BLEU scores by decoding strategy
    decoding_scores = {}
    for strategy in decoding_strategies:
        scores = [r['bleu_score'] for r in results_data if r['decoding'] == strategy]
        decoding_scores[strategy] = np.mean(scores) if scores else 0

    colors = ['lightgreen', 'gold', 'lightpink'][:len(decoding_scores)]
    ax2.bar(decoding_scores.keys(), decoding_scores.values(), color=colors)
    ax2.set_ylabel('Average BLEU Score')
    ax2.set_title('BLEU Scores by Decoding Strategy')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (k, v) in enumerate(decoding_scores.items()):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/bleu_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_translation_examples():
    """Show some translation examples from the best performing model"""
    results_dir = Path('results')
    result_files = list(results_dir.glob('results_*.json'))

    if not result_files:
        print("No result files found.")
        return

    # Find best performing model
    best_result = None
    best_bleu = 0

    for file in result_files:
        try:
            result = load_results(file)
            if result.get('bleu_score', 0) > best_bleu:
                best_bleu = result.get('bleu_score', 0)
                best_result = result
        except Exception:
            continue

    if not best_result:
        print("No valid results found.")
        return

    print("\n" + "="*80)
    print(f"TRANSLATION EXAMPLES - BEST MODEL")
    print(f"Model: {best_result.get('positional_encoding')} + {best_result.get('decoding_strategy')}")
    print(f"BLEU Score: {best_result.get('bleu_score', 0):.4f}")
    print("="*80)

    examples = best_result.get('examples', [])
    for i, example in enumerate(examples[:5]):  # Show first 5 examples
        print(f"\nExample {i+1}:")
        print(f"Source (FI): {example.get('source', '')}")
        print(f"Reference (EN): {example.get('reference', '')}")
        print(f"Translation: {example.get('translation', '')}")
        print("-" * 60)

def main():
    """Main analysis function"""
    print("TRANSFORMER MACHINE TRANSLATION - RESULTS ANALYSIS")
    print("=" * 60)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Plot training curves comparison
    print("\n1. Training Curves Analysis:")
    plot_training_curves_comparison()

    # Create BLEU scores table
    print("\n2. BLEU Scores Analysis:")
    create_bleu_scores_table()

    # Plot BLEU comparison
    print("\n3. BLEU Scores Visualization:")
    plot_bleu_comparison()

    # Show translation examples
    print("\n4. Translation Examples:")
    show_translation_examples()

    print(f"\nAnalysis complete! Check the 'results/' directory for saved plots.")

if __name__ == '__main__':
    main()
