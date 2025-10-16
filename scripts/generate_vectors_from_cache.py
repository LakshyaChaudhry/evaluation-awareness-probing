#!/usr/bin/env python
"""
Script for generating contrastive steering vectors from cached activations.

This script loads pre-computed activations instead of running model inference,
making probe training much faster for repeated experiments.

Usage:
    python generate_vectors_from_cache.py --cache activations_cache/llama70b --output run_70b_experiment1
"""
import argparse
import os
import json
import torch
from datetime import datetime

def load_cached_activations(cache_dir, layers=None):
    """
    Load cached activations from disk.

    Args:
        cache_dir: Directory containing cached activations
        layers: Optional list of layers to load (loads all if None)

    Returns:
        tuple: (test_activations_dict, deploy_activations_dict, metadata, layer_info)
    """
    # Load metadata and layer info
    with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    with open(os.path.join(cache_dir, 'layer_info.json'), 'r') as f:
        layer_info = json.load(f)

    available_layers = layer_info['layers']
    if layers is None:
        layers = available_layers
    else:
        # Verify requested layers are available
        missing = [l for l in layers if l not in available_layers]
        if missing:
            raise ValueError(f"Requested layers {missing} not found in cache. Available: {available_layers}")

    print(f"Loading activations for {len(layers)} layers: {layers}")

    # Load activations per layer
    test_activations = {}
    deploy_activations = {}

    for layer in layers:
        layer_dir = os.path.join(cache_dir, f'layer_{layer}')
        test_acts = torch.load(os.path.join(layer_dir, 'test_activations.pt'))
        deploy_acts = torch.load(os.path.join(layer_dir, 'deploy_activations.pt'))

        test_activations[layer] = test_acts
        deploy_activations[layer] = deploy_acts

        print(f"Loaded layer {layer}: test {test_acts.shape}, deploy {deploy_acts.shape}")

    return test_activations, deploy_activations, metadata, layer_info

def compute_steering_vectors(test_activations, deploy_activations, layers):
    """
    Compute steering vectors from cached activations.

    Args:
        test_activations: Dict of layer -> test activation tensors
        deploy_activations: Dict of layer -> deploy activation tensors
        layers: List of layer indices

    Returns:
        dict: Dictionary mapping layer indices to steering vectors
    """
    steering_vectors = {}

    for layer in layers:
        # Compute mean activations
        test_mean = test_activations[layer].mean(dim=0)
        deploy_mean = deploy_activations[layer].mean(dim=0)

        # Steering vector = test_mean - deploy_mean
        steering_vector = test_mean - deploy_mean
        steering_vectors[layer] = steering_vector

        print(f"Layer {layer}: steering vector shape {steering_vector.shape}, norm {steering_vector.norm().item():.4f}")

    return steering_vectors

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    return vector * (1.0 / vector.norm())

def save_vectors(steering_vectors, output_dir):
    """
    Save steering vectors to disk.

    Args:
        steering_vectors: Dict of layer -> vector
        output_dir: Base output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"{output_dir}_{timestamp}"

    # Create directories
    vectors_dir = os.path.join(base_dir, 'vectors')
    norm_vectors_dir = os.path.join(base_dir, 'normalized_vectors')
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(norm_vectors_dir, exist_ok=True)

    # Save vectors
    for layer, vector in steering_vectors.items():
        # Save raw vector
        raw_path = os.path.join(vectors_dir, f"layer_{layer}.pt")
        torch.save(vector, raw_path)

        # Save normalized vector
        normalized = normalize_vector(vector)
        norm_path = os.path.join(norm_vectors_dir, f"layer_{layer}.pt")
        torch.save(normalized, norm_path)

        print(f"Saved layer {layer}: raw -> {raw_path}, normalized -> {norm_path}")

    # Save experiment info
    with open(os.path.join(base_dir, 'experiment_info.txt'), 'w') as f:
        f.write(f"Generated from cached activations\n")
        f.write(f"Layers: {list(steering_vectors.keys())}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print(f"\n✓ Vectors saved to {base_dir}")
    return base_dir

def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors from cached activations")
    parser.add_argument("--cache", required=True, help="Directory with cached activations")
    parser.add_argument("--output", required=True, help="Output directory name prefix")
    parser.add_argument("--layers", help="Comma-separated list of layers (default: all cached layers)")
    args = parser.parse_args()

    # Parse layers if specified
    layers = None
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
        print(f"Requested layers: {layers}")

    # Load cached activations
    print(f"\nLoading cached activations from: {args.cache}")
    test_acts, deploy_acts, metadata, layer_info = load_cached_activations(args.cache, layers)

    if layers is None:
        layers = layer_info['layers']

    print(f"\nProcessing {layer_info['num_examples']} cached examples")

    # Compute steering vectors
    print(f"\nComputing steering vectors for {len(layers)} layers...")
    steering_vectors = compute_steering_vectors(test_acts, deploy_acts, layers)

    # Save vectors
    print(f"\nSaving vectors...")
    output_dir = save_vectors(steering_vectors, args.output)

    print(f"\n✓ Experiment completed!")
    print(f"  Output: {output_dir}")
    print(f"  Layers: {layers}")
    print(f"\nTo validate these probes, run:")
    print(f"  python scripts/analyze_probe.py \\")
    print(f"    --model <MODEL_PATH> \\")
    print(f"    --vectors {output_dir}/normalized_vectors \\")
    print(f"    --data datasets/validation_real.json \\")
    print(f"    --output validation_results \\")
    print(f"    --visualize")

if __name__ == "__main__":
    main()
