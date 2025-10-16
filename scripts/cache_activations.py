#!/usr/bin/env python
"""
Script for extracting and caching model activations.

This script extracts activations from specified layers and saves them to disk,
allowing for faster experimentation without re-running expensive model inference.

Usage:
    python cache_activations.py --model MODEL_PATH --data DATASET_PATH --layers 10,20,23,25,30,40,50,60,70 --output activations_cache/llama70b
"""
import argparse
import os
import json
import torch
from tqdm import tqdm
import gc

import sys
sys.path.append('.')
from src.utils import load_model, load_contrastive_dataset

def extract_activations_batch(model, prompts, layers, tokenizer):
    """
    Extract activations for a batch of prompts at specified layers.

    Args:
        model: HookedTransformer model
        prompts: List of text prompts
        layers: List of layer indices to extract from
        tokenizer: Tokenizer

    Returns:
        list: List of dicts mapping layer -> activation tensor for each prompt
    """
    all_activations = []

    for prompt in tqdm(prompts, desc="Extracting activations"):
        tokens = model.to_tokens(prompt)
        cached_activations = {}

        def make_hook(layer):
            def hook_fn(activation, hook):
                # Store the last token's activation
                cached_activations[layer] = activation[:, -1, :].clone().detach().cpu()
                return activation
            return hook_fn

        # Set up hooks for all layers
        fwd_hooks = []
        for layer in layers:
            hook = make_hook(layer)
            hook_name = f"blocks.{layer}.hook_resid_pre"
            fwd_hooks.append((hook_name, hook))

        # Run forward pass with hooks
        model.reset_hooks()
        with model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                model(tokens)

        all_activations.append(cached_activations)

        # Clean up
        del tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_activations

def cache_dataset_activations(model, dataset, layers, output_dir, dataset_name, tokenizer):
    """
    Cache activations for an entire dataset.

    Args:
        model: HookedTransformer model
        dataset: Dataset (list of dicts with 'question', 'test', 'deploy')
        layers: List of layer indices
        output_dir: Directory to save cached activations
        dataset_name: Name identifier for this dataset
        tokenizer: Tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare all prompts (both test and deploy variants)
    test_prompts = []
    deploy_prompts = []
    metadata = []

    print(f"Preparing {len(dataset)} examples...")
    for idx, item in enumerate(dataset):
        question = item["question"]
        test_answer = item['test']
        deploy_answer = item['deploy']

        # Build prompts
        test_prompt = f"{question}\n\n {test_answer}"
        deploy_prompt = f"{question}\n\n {deploy_answer}"

        test_prompts.append(test_prompt)
        deploy_prompts.append(deploy_prompt)

        # Store metadata
        metadata.append({
            'idx': idx,
            'question': question,
            'test_answer': test_answer,
            'deploy_answer': deploy_answer,
            'test_token_type': test_answer[1],  # e.g., 'A' from '(A)'
            'deploy_token_type': deploy_answer[1]
        })

    # Extract activations for test prompts
    print(f"\nExtracting TEST activations for {len(test_prompts)} prompts across {len(layers)} layers...")
    test_activations = extract_activations_batch(model, test_prompts, layers, tokenizer)

    # Extract activations for deploy prompts
    print(f"\nExtracting DEPLOY activations for {len(deploy_prompts)} prompts across {len(layers)} layers...")
    deploy_activations = extract_activations_batch(model, deploy_prompts, layers, tokenizer)

    # Save activations and metadata
    print(f"\nSaving to {output_dir}...")

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Save layer info
    layer_info = {
        'layers': layers,
        'num_examples': len(dataset),
        'activation_shape': test_activations[0][layers[0]].shape
    }
    layer_info_path = os.path.join(output_dir, 'layer_info.json')
    with open(layer_info_path, 'w') as f:
        json.dump(layer_info, f, indent=2)
    print(f"Saved layer info: {layer_info_path}")

    # Save activations per layer (more memory efficient for loading)
    for layer in layers:
        # Stack all test activations for this layer
        test_acts_layer = torch.stack([act[layer] for act in test_activations])
        deploy_acts_layer = torch.stack([act[layer] for act in deploy_activations])

        layer_dir = os.path.join(output_dir, f'layer_{layer}')
        os.makedirs(layer_dir, exist_ok=True)

        torch.save(test_acts_layer, os.path.join(layer_dir, 'test_activations.pt'))
        torch.save(deploy_acts_layer, os.path.join(layer_dir, 'deploy_activations.pt'))

        print(f"Saved layer {layer}: test shape {test_acts_layer.shape}, deploy shape {deploy_acts_layer.shape}")

    print(f"\n✓ Successfully cached activations for {len(dataset)} examples across {len(layers)} layers")
    print(f"  Cache directory: {output_dir}")
    print(f"  Disk usage: ~{estimate_cache_size(layers[0], len(dataset), test_acts_layer.shape[1], len(layers))}")

def estimate_cache_size(layer, num_examples, hidden_dim, num_layers):
    """Estimate cache size in GB."""
    bytes_per_activation = hidden_dim * 4  # float32
    total_bytes = bytes_per_activation * num_examples * 2 * num_layers  # test + deploy
    gb = total_bytes / (1024**3)
    return f"{gb:.2f} GB"

def main():
    parser = argparse.ArgumentParser(description="Cache model activations for faster experimentation")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--data", required=True, help="Path to dataset (contrastive or validation)")
    parser.add_argument("--layers", required=True, help="Comma-separated list of layers to cache")
    parser.add_argument("--output", required=True, help="Output directory for cached activations")
    parser.add_argument("--device", default="cuda", help="Device to run on (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", help="Data type (default: bfloat16)")
    parser.add_argument("--dataset-name", default="dataset", help="Name identifier for this dataset")
    args = parser.parse_args()

    # Set dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Parse layers
    layers = [int(l) for l in args.layers.split(",")]
    print(f"Caching layers: {layers}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)
    print(f"Model loaded on device: {model.cfg.device if hasattr(model.cfg, 'device') else 'multiple GPUs'}")

    # Load dataset
    print(f"\nLoading dataset: {args.data}")
    dataset = load_contrastive_dataset(args.data, tokenizer)
    print(f"Loaded {len(dataset)} examples")

    # Cache activations
    cache_dataset_activations(
        model=model,
        dataset=dataset,
        layers=layers,
        output_dir=args.output,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()
