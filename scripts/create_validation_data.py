#!/usr/bin/env python
"""
Create proper validation dataset with real eval and deploy data.
Downloads GSM8K (evaluation) and Alpaca (deployment) datasets.

Usage:
    python scripts/create_validation_data.py --output datasets/validation_real.json --num-examples 100
"""
import argparse
import json
import random

def main():
    parser = argparse.ArgumentParser(description="Create validation dataset")
    parser.add_argument("--output", default="datasets/validation_real.json", help="Output file path")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples per type (default: 100)")
    args = parser.parse_args()

    print("Downloading datasets...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found.")
        print("Install with: pip install datasets")
        return

    validation_data = []

    # Load GSM8K (evaluation/testing data)
    print(f"Loading GSM8K (evaluation data)...")
    try:
        gsm8k = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        # Sample random examples
        indices = random.sample(range(len(gsm8k)), min(args.num_examples, len(gsm8k)))
        for idx in indices:
            ex = gsm8k[idx]
            validation_data.append({
                "question": ex["question"],
                "type": "testing"
            })
        print(f"  Added {len(indices)} GSM8K examples")
    except Exception as e:
        print(f"  WARNING: Could not load GSM8K: {e}")
        print("  Skipping evaluation data...")

    # Load Alpaca (deployment data)
    print(f"Loading Alpaca (deployment data)...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        # Sample random examples
        indices = random.sample(range(len(alpaca)), min(args.num_examples, len(alpaca)))
        for idx in indices:
            ex = alpaca[idx]
            # Use instruction field
            question = ex.get("instruction", "")
            if ex.get("input"):  # Append input if it exists
                question += f"\n\n{ex['input']}"

            validation_data.append({
                "question": question,
                "type": "deployment"
            })
        print(f"  Added {len(indices)} Alpaca examples")
    except Exception as e:
        print(f"  WARNING: Could not load Alpaca: {e}")
        print("  Skipping deployment data...")

    if not validation_data:
        print("ERROR: No data was loaded. Cannot create validation dataset.")
        return

    # Shuffle to mix testing and deployment examples
    random.shuffle(validation_data)

    # Save
    with open(args.output, 'w') as f:
        json.dump(validation_data, f, indent=2)

    # Report statistics
    testing_count = sum(1 for x in validation_data if x['type'] == 'testing')
    deployment_count = sum(1 for x in validation_data if x['type'] == 'deployment')

    print(f"\n✓ Created validation dataset: {args.output}")
    print(f"  Total: {len(validation_data)} examples")
    print(f"  Testing: {testing_count}")
    print(f"  Deployment: {deployment_count}")

if __name__ == "__main__":
    main()
