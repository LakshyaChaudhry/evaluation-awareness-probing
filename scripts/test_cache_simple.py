#!/usr/bin/env python
"""
Quick test script to validate activation caching works correctly.

Tests on a small subset of data to verify the caching mechanism.

Usage:
    python test_cache_simple.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import os
import json
import torch
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append('.')
from src.utils import load_model, load_contrastive_dataset

def test_caching(model_name, num_examples=10):
    """
    Test that caching and loading activations works correctly.

    Args:
        model_name: Model to test with
        num_examples: Number of examples to test (small for speed)
    """
    print("="*60)
    print("TESTING ACTIVATION CACHING")
    print("="*60)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="cache_test_")
    cache_dir = os.path.join(temp_dir, "test_cache")

    try:
        # Load model
        print(f"\n1. Loading model: {model_name}")
        model, tokenizer = load_model(model_name, device="cuda", dtype=torch.bfloat16)
        print("✓ Model loaded")

        # Load small dataset
        print(f"\n2. Loading dataset (first {num_examples} examples)")
        dataset = load_contrastive_dataset("datasets/contrastive_dataset.json", tokenizer)
        dataset = dataset[:num_examples]  # Use only first N examples
        print(f"✓ Loaded {len(dataset)} examples")

        # Extract and cache activations
        print("\n3. Extracting activations directly from model")
        layers = [15, 20, 25]

        test_activations_direct = {layer: [] for layer in layers}
        deploy_activations_direct = {layer: [] for layer in layers}

        for item in dataset:
            question = item["question"]
            test_answer = item['test']
            deploy_answer = item['deploy']

            test_prompt = f"{question}\n\n {test_answer}"
            deploy_prompt = f"{question}\n\n {deploy_answer}"

            # Get activations for test prompt
            test_tokens = model.to_tokens(test_prompt)
            test_acts = {}

            def make_test_hook(layer_idx):
                def hook_fn(activation, hook):
                    test_acts[layer_idx] = activation[:, -1, :].clone().detach().cpu()
                    return activation
                return hook_fn

            fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", make_test_hook(layer)) for layer in layers]
            model.reset_hooks()
            with model.hooks(fwd_hooks=fwd_hooks):
                with torch.no_grad():
                    model(test_tokens)

            for layer in layers:
                test_activations_direct[layer].append(test_acts[layer])

            # Get activations for deploy prompt
            deploy_tokens = model.to_tokens(deploy_prompt)
            deploy_acts = {}

            def make_deploy_hook(layer_idx):
                def hook_fn(activation, hook):
                    deploy_acts[layer_idx] = activation[:, -1, :].clone().detach().cpu()
                    return activation
                return hook_fn

            fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", make_deploy_hook(layer)) for layer in layers]
            model.reset_hooks()
            with model.hooks(fwd_hooks=fwd_hooks):
                with torch.no_grad():
                    model(deploy_tokens)

            for layer in layers:
                deploy_activations_direct[layer].append(deploy_acts[layer])

        # Stack activations
        for layer in layers:
            test_activations_direct[layer] = torch.stack(test_activations_direct[layer])
            deploy_activations_direct[layer] = torch.stack(deploy_activations_direct[layer])

        print(f"✓ Extracted activations for {len(layers)} layers")

        # Save to cache
        print("\n4. Saving activations to cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Save metadata
        metadata = [{'idx': i, 'question': item['question'][:50]} for i, item in enumerate(dataset)]
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # Save layer info
        layer_info = {
            'layers': layers,
            'num_examples': len(dataset),
            'activation_shape': list(test_activations_direct[layers[0]][0].shape)
        }
        with open(os.path.join(cache_dir, 'layer_info.json'), 'w') as f:
            json.dump(layer_info, f)

        # Save activations
        for layer in layers:
            layer_dir = os.path.join(cache_dir, f'layer_{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            torch.save(test_activations_direct[layer], os.path.join(layer_dir, 'test_activations.pt'))
            torch.save(deploy_activations_direct[layer], os.path.join(layer_dir, 'deploy_activations.pt'))

        print(f"✓ Saved to {cache_dir}")

        # Load from cache
        print("\n5. Loading activations from cache")
        test_activations_cached = {}
        deploy_activations_cached = {}

        for layer in layers:
            layer_dir = os.path.join(cache_dir, f'layer_{layer}')
            test_activations_cached[layer] = torch.load(os.path.join(layer_dir, 'test_activations.pt'))
            deploy_activations_cached[layer] = torch.load(os.path.join(layer_dir, 'deploy_activations.pt'))

        print(f"✓ Loaded {len(layers)} layers from cache")

        # Compare activations
        print("\n6. Comparing direct vs cached activations")
        all_match = True

        for layer in layers:
            test_match = torch.allclose(test_activations_direct[layer], test_activations_cached[layer], rtol=1e-5)
            deploy_match = torch.allclose(deploy_activations_direct[layer], deploy_activations_cached[layer], rtol=1e-5)

            if test_match and deploy_match:
                print(f"✓ Layer {layer}: Activations match perfectly")
            else:
                print(f"✗ Layer {layer}: Activations DO NOT match")
                all_match = False

                if not test_match:
                    diff = (test_activations_direct[layer] - test_activations_cached[layer]).abs().max()
                    print(f"  Test max diff: {diff}")
                if not deploy_match:
                    diff = (deploy_activations_direct[layer] - deploy_activations_cached[layer]).abs().max()
                    print(f"  Deploy max diff: {diff}")

        # Test probe computation
        print("\n7. Testing probe computation from cached activations")
        for layer in layers:
            test_mean = test_activations_cached[layer].mean(dim=0)
            deploy_mean = deploy_activations_cached[layer].mean(dim=0)
            probe = test_mean - deploy_mean

            print(f"✓ Layer {layer}: Probe shape {probe.shape}, norm {probe.norm().item():.4f}")

        # Final result
        print("\n" + "="*60)
        if all_match:
            print("✓✓✓ TEST PASSED ✓✓✓")
            print("Caching mechanism works correctly!")
            print("You can now use the cache scripts with confidence.")
        else:
            print("✗✗✗ TEST FAILED ✗✗✗")
            print("Activations from cache don't match direct extraction.")
        print("="*60)

    finally:
        # Cleanup
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Test activation caching")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model to test with")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to test (default: 10)")
    args = parser.parse_args()

    test_caching(args.model, args.num_examples)

if __name__ == "__main__":
    main()
