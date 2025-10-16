#!/usr/bin/env python
"""
Script for analyzing probe performance using cached activations.

This script loads pre-computed activations and evaluates probes without
running model inference, enabling fast experimentation.

Usage:
    python analyze_probe_from_cache.py --cache activations_cache/llama70b_validation --vectors run_70b/normalized_vectors --output validation_results
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

def load_cached_activations(cache_dir, layers=None):
    """
    Load cached activations from disk.

    Args:
        cache_dir: Directory containing cached activations
        layers: Optional list of layers to load (loads all if None)

    Returns:
        tuple: (activations_dict, metadata, layer_info)
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
    # We need to determine the "type" for each example from metadata
    # and then load the corresponding activation (test or deploy)
    activations = {}

    for layer in layers:
        layer_dir = os.path.join(cache_dir, f'layer_{layer}')
        test_acts = torch.load(os.path.join(layer_dir, 'test_activations.pt'))
        deploy_acts = torch.load(os.path.join(layer_dir, 'deploy_activations.pt'))

        # Store both - we'll select the right one based on the type field
        activations[layer] = {
            'test': test_acts,
            'deploy': deploy_acts
        }

        print(f"Loaded layer {layer}: test {test_acts.shape}, deploy {deploy_acts.shape}")

    return activations, metadata, layer_info

def load_vectors_from_dir(directory):
    """
    Load vectors from a directory.

    Args:
        directory: Directory containing vectors

    Returns:
        dict: Dictionary of vectors by layer
    """
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

    Args:
        activation: Activation tensor
        probe_vector: Probe vector

    Returns:
        float: Probe score
    """
    # Flatten both tensors to 1D
    activation_flat = activation.flatten()
    probe_flat = probe_vector.flatten()

    # Project activation onto probe vector
    # Score = dot(activation, normalized_probe)
    score = torch.dot(activation_flat, probe_flat) / probe_flat.norm()
    return score.item()

def analyze_cached_dataset(activations, metadata, probe_vectors, layer, output_file=None):
    """
    Analyze a dataset using cached activations and probes.

    Args:
        activations: Dict of layer -> {'test': tensor, 'deploy': tensor}
        metadata: List of metadata dicts for each example
        probe_vectors: Dictionary of probe vectors by layer
        layer: Layer to analyze
        output_file: Path to save results

    Returns:
        list: Results for each example
    """
    if layer not in activations:
        raise ValueError(f"Layer {layer} not found in cached activations")

    if layer not in probe_vectors:
        raise ValueError(f"Layer {layer} not found in probe vectors")

    probe = probe_vectors[layer]
    layer_acts = activations[layer]

    results = []

    print(f"Analyzing {len(metadata)} examples for layer {layer}...")

    for idx, item in enumerate(tqdm(metadata, desc="Processing examples")):
        # Determine which activation to use
        # The validation dataset should have a 'type' field in metadata
        # But the cached activations were created from contrastive data
        # So we need to handle both test and deploy contexts

        # For validation data: we have separate test/deploy prompts for each question
        # We'll score both and store them separately
        # Actually, looking at the metadata structure, each item corresponds to ONE prompt
        # So we need to figure out which type it is

        # Let's check if there's a 'type' field in the original validation data
        # For now, we'll score the test activation (last token)

        test_score = compute_probe_score(layer_acts['test'][idx], probe)
        deploy_score = compute_probe_score(layer_acts['deploy'][idx], probe)

        # For validation, we want to score based on the type
        # The metadata tells us if this is a testing or deployment example
        # But wait - the cache was built with BOTH test and deploy for each question
        # So we have 2 activations per question

        # We need to determine: is this a "testing" or "deployment" type question?
        # That info should be in a separate validation metadata file

        results.append({
            'idx': idx,
            'question': item['question'][:100] + '...',  # truncate for readability
            'test_score': test_score,
            'deploy_score': deploy_score,
        })

    # Save results
    if output_file:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return results

def analyze_validation_from_cache(cache_dir, validation_data_path, probe_vectors, layer, output_file=None):
    """
    Analyze validation data using cached activations.

    This is a specialized function that properly handles validation data
    where we need to match examples with their types.

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

    # Load cached activations
    with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
        cache_metadata = json.load(f)

    # Load layer activations
    layer_dir = os.path.join(cache_dir, f'layer_{layer}')
    test_acts = torch.load(os.path.join(layer_dir, 'test_activations.pt'))
    deploy_acts = torch.load(os.path.join(layer_dir, 'deploy_activations.pt'))

    probe = probe_vectors[layer]

    results = []

    # For each validation example, we have a type (testing/deployment)
    # We need to match it with the corresponding cached activation
    print(f"Processing {len(validation_data)} validation examples...")

    for idx in tqdm(range(len(validation_data))):
        val_item = validation_data[idx]
        item_type = val_item.get('type', 'unknown')

        # The cache has both test and deploy activations for each question
        # We need to use the one that matches the validation type

        # For "testing" type: use test activation
        # For "deployment" type: use deploy activation
        if item_type == 'testing':
            activation = test_acts[idx]
        elif item_type == 'deployment':
            activation = deploy_acts[idx]
        else:
            print(f"Warning: Unknown type '{item_type}' for example {idx}")
            continue

        # Compute probe score
        score = compute_probe_score(activation, probe)

        results.append({
            'question_id': idx,
            'question': val_item.get('question', '')[:100] + '...',
            'type': item_type,
            'mean_score': score,  # Use score as mean_score to match analyze_probe.py format
            'max_score': score,
            'min_score': score,
            'std_score': 0.0
        })

    # Save results
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Print statistics
        testing_scores = df[df['type'] == 'testing']['mean_score']
        deployment_scores = df[df['type'] == 'deployment']['mean_score']

        print(f"\nLayer {layer} Statistics:")
        print(f"  Testing examples: {len(testing_scores)}, mean score: {testing_scores.mean():.4f}")
        print(f"  Deployment examples: {len(deployment_scores)}, mean score: {deployment_scores.mean():.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze probes using cached activations")
    parser.add_argument("--cache", required=True, help="Directory with cached activations")
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

        analyze_validation_from_cache(
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
