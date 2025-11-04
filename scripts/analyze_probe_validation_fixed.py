#!/usr/bin/env python
"""
Analyze probes using correctly cached validation activations.

This version works with validation data that has ONE activation per example
(from raw questions, not appended with (A) or (B)).

Usage:
    python analyze_probe_validation_fixed.py --cache activations_cache/hal9000_validation_fixed --vectors run_hal9000_TIMESTAMP/normalized_vectors --data datasets/validation_real.json --output validation_hal9000_fixed --visualize
"""
import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('.')
from src.visualization import plot_all_rocs_in_directory

def load_vectors_from_dir(directory):
    """Load probe vectors from directory."""
    vectors = {}
    for filename in os.listdir(directory):
        if filename.startswith('layer_') and filename.endswith('.pt'):
            layer = int(filename.split('_')[1].split('.')[0])
            vectors[layer] = torch.load(os.path.join(directory, filename))
            print(f"Loaded probe for layer {layer}")
    return vectors

def compute_probe_score(activation, probe_vector):
    """
    Compute probe score for an activation.

    Following the paper: average dot product across all tokens.

    Args:
        activation: Tensor of shape [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        probe_vector: Probe vector of shape [hidden_dim]

    Returns:
        float: Averaged probe score
    """
    # Handle multi-token activations
    if activation.dim() == 3:  # [batch, seq_len, hidden_dim]
        # Compute dot product for each token, then average
        # activation shape: [1, seq_len, hidden_dim]
        # probe shape: [hidden_dim]
        activation = activation.squeeze(0)  # [seq_len, hidden_dim]

        # Compute dot product for each token
        token_scores = torch.matmul(activation, probe_vector) / probe_vector.norm()  # [seq_len]

        # Average across tokens (as per the paper)
        score = token_scores.mean()
        return score.item()
    else:
        # Old format: [batch, hidden_dim] or [hidden_dim]
        activation_flat = activation.flatten()
        probe_flat = probe_vector.flatten()
        score = torch.dot(activation_flat, probe_flat) / probe_flat.norm()
        return score.item()

def analyze_validation(cache_dir, validation_data_path, probe_vectors, layer, output_file=None):
    """
    Analyze validation data using correctly cached activations.

    Args:
        cache_dir: Directory with cached activations
        validation_data_path: Path to validation JSON with type labels
        probe_vectors: Dictionary of probe vectors
        layer: Layer to analyze
        output_file: Path to save results
    """
    # Load validation data to get types
    with open(validation_data_path, 'r') as f:
        validation_data = json.load(f)

    # Load cached activations for this layer
    layer_dir = os.path.join(cache_dir, f'layer_{layer}')

    # Check if using new individual storage format or old stacked format
    individual_file = os.path.join(layer_dir, '0.pt')
    stacked_file = os.path.join(layer_dir, 'activations.pt')

    if os.path.exists(individual_file):
        # New format: individual files
        print(f"Loading individual activation files for layer {layer}...")
        activations = []
        for idx in range(len(validation_data)):
            act = torch.load(os.path.join(layer_dir, f'{idx}.pt'))
            activations.append(act)
    elif os.path.exists(stacked_file):
        # Old format: single stacked file
        print(f"Loading stacked activations for layer {layer}...")
        activations = torch.load(stacked_file)
        # Convert to list for uniform handling
        activations = [activations[i] for i in range(len(activations))]
    else:
        raise FileNotFoundError(f"Could not find activations in {layer_dir}")

    probe = probe_vectors[layer]

    results = []

    print(f"Processing {len(validation_data)} validation examples for layer {layer}...")

    for idx in tqdm(range(len(validation_data))):
        val_item = validation_data[idx]
        item_type = val_item.get('type', 'unknown')

        # Get the activation for this example
        activation = activations[idx]

        # Compute probe score (will average over tokens if needed)
        score = compute_probe_score(activation, probe)

        results.append({
            'question_id': idx,
            'question': val_item.get('question', '')[:100] + '...',
            'type': item_type,
            'mean_score': score,
            'max_score': score,
            'min_score': score,
            'std_score': 0.0
        })

    # Save results
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to {output_file}")

        # Print statistics
        testing_scores = df[df['type'] == 'testing']['mean_score']
        deployment_scores = df[df['type'] == 'deployment']['mean_score']

        print(f"\nLayer {layer} Statistics:")
        print(f"  Testing: {len(testing_scores)} examples, mean={testing_scores.mean():.4f}, std={testing_scores.std():.4f}")
        print(f"  Deployment: {len(deployment_scores)} examples, mean={deployment_scores.mean():.4f}, std={deployment_scores.std():.4f}")
        print(f"  Difference: {deployment_scores.mean() - testing_scores.mean():.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze probes with fixed validation data")
    parser.add_argument("--cache", required=True, help="Directory with cached validation activations")
    parser.add_argument("--vectors", required=True, help="Directory with probe vectors")
    parser.add_argument("--data", required=True, help="Path to validation dataset (with type labels)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--layers", help="Comma-separated list of layers to analyze")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--show-plots", action="store_true", help="Show plots (requires display)")
    args = parser.parse_args()

    # Load probe vectors
    print(f"Loading probe vectors from: {args.vectors}")
    probe_vectors = load_vectors_from_dir(args.vectors)

    # Determine layers to analyze
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = list(probe_vectors.keys())

    print(f"Analyzing layers: {layers}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Analyze each layer
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Analyzing layer {layer}")
        print(f"{'='*60}")

        output_file = os.path.join(args.output, f"probe_stats_layer{layer}.csv")

        analyze_validation(
            cache_dir=args.cache,
            validation_data_path=args.data,
            probe_vectors=probe_vectors,
            layer=layer,
            output_file=output_file
        )

    # Generate visualizations if requested
    if args.visualize:
        print(f"\n{'='*60}")
        print("Generating visualizations...")
        print(f"{'='*60}")

        viz_dir = os.path.join(args.output, "visualizations")
        metrics = plot_all_rocs_in_directory(args.output, viz_dir, show=args.show_plots)

    print(f"\n✓ Analysis completed!")
    print(f"  Results saved to: {args.output}")

    # Print AUROC summary
    print(f"\n{'='*60}")
    print("AUROC Summary:")
    print(f"{'='*60}")

    from src.probe_analysis import process_directory
    process_directory(args.output, verbose=True)

if __name__ == "__main__":
    main()
