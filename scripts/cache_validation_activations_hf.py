#!/usr/bin/env python
"""
Script for caching validation activations correctly.

This version extracts activations from the RAW questions without appending (A) or (B),
ensuring we're testing actual evaluation awareness, not token detection.

Usage:
    python cache_validation_activations_hf.py --model bench-af/paranoid-hal9000 --data datasets/validation_real.json --layers 10,20,23,25,30,40,50,60,70 --output activations_cache/hal9000_validation_fixed
"""
import argparse
import os
import json
import torch
from tqdm import tqdm
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_hf(model_path, dtype=torch.bfloat16):
    """Load HuggingFace model directly."""
    print(f"Loading HuggingFace model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✓ Model loaded")

    return model, tokenizer

def extract_activation_hf(model, tokens, layer_idx):
    """
    Extract activation from a specific layer using HuggingFace model.

    Args:
        model: HuggingFace model
        tokens: Input token IDs
        layer_idx: Layer index to extract from

    Returns:
        Activation tensor for ALL tokens (shape: [batch_size, seq_len, hidden_dim])
    """
    activation = None

    def hook_fn(module, input, output):
        nonlocal activation
        # output[0] is the hidden states
        # Take ALL token positions, not just the last one
        activation = output[0].detach().cpu().clone()  # [batch, seq_len, hidden]

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(tokens)
    finally:
        hook_handle.remove()

    return activation

def load_validation_dataset(file_path):
    """Load validation dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def cache_validation_activations(model, tokenizer, dataset, layers, output_dir):
    """
    Cache activations for validation dataset.

    KEY DIFFERENCE: We extract ONE activation per example from the RAW question,
    NOT by appending (A) or (B).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize storage - ONE activation per example
    activations_by_layer = {layer: [] for layer in layers}
    metadata = []

    print(f"\nProcessing {len(dataset)} validation examples across {len(layers)} layers...")
    print("Extracting activations from RAW questions (no (A)/(B) tokens added)")

    for idx, item in enumerate(tqdm(dataset, desc="Extracting activations")):
        question = item['question']
        item_type = item['type']  # 'testing' or 'deployment'

        # Use the RAW question - no (A) or (B) appended!
        tokens = tokenizer(question, return_tensors="pt").input_ids
        device = next(model.parameters()).device
        tokens = tokens.to(device)

        # Extract activation for each layer
        for layer in layers:
            activation = extract_activation_hf(model, tokens, layer)
            activations_by_layer[layer].append(activation)

        # Store metadata
        metadata.append({
            'idx': idx,
            'question': question[:100] + '...',  # Truncate for readability
            'type': item_type
        })

        # Clean up
        del tokens
        if (idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save metadata
    print("\nSaving activations...")
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    # Get sample activation shapes (first 3 examples to show variability)
    sample_shapes = [activations_by_layer[layers[0]][i].shape for i in range(min(3, len(dataset)))]

    # Save layer info
    layer_info = {
        'layers': layers,
        'num_examples': len(dataset),
        'storage_format': 'individual',  # Indicate storage format
        'sample_shapes': [list(s) for s in sample_shapes],  # Show example shapes
        'note': 'Validation activations - one per example, from raw questions, variable seq lengths'
    }
    layer_info_path = os.path.join(output_dir, 'layer_info.json')
    with open(layer_info_path, 'w') as f:
        json.dump(layer_info, f, indent=2)
    print(f"✓ Saved layer info: {layer_info_path}")

    # Save activations per layer
    for layer in layers:
        layer_dir = os.path.join(output_dir, f'layer_{layer}')
        os.makedirs(layer_dir, exist_ok=True)

        # Save each activation individually to handle variable sequence lengths
        for idx, activation in enumerate(activations_by_layer[layer]):
            torch.save(activation, os.path.join(layer_dir, f'{idx}.pt'))

        # Show sample shapes to verify variable lengths
        sample_shapes = [activations_by_layer[layer][i].shape for i in range(min(3, len(activations_by_layer[layer])))]
        print(f"✓ Layer {layer}: saved {len(activations_by_layer[layer])} activations")
        print(f"  Sample shapes: {sample_shapes}")

    print(f"\n✓ Successfully cached {len(dataset)} validation activations across {len(layers)} layers")
    print(f"  Cache directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Cache validation activations (fixed version)")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--data", required=True, help="Path to validation dataset")
    parser.add_argument("--layers", required=True, help="Comma-separated list of layers")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dtype", default="bfloat16", help="Data type (default: bfloat16)")
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
    model, tokenizer = load_model_hf(args.model, dtype=dtype)

    # Load validation dataset
    print(f"\nLoading validation dataset: {args.data}")
    dataset = load_validation_dataset(args.data)
    print(f"✓ Loaded {len(dataset)} examples")

    # Verify dataset format
    testing_count = sum(1 for item in dataset if item['type'] == 'testing')
    deployment_count = sum(1 for item in dataset if item['type'] == 'deployment')
    print(f"  Testing: {testing_count}, Deployment: {deployment_count}")

    # Cache activations
    cache_validation_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
