# Activation Caching Workflow

This guide shows how to use pre-computed activations for faster experimentation with 70B models.

## Why Use Activation Caching?

When running multiple experiments on the same model:
- **Without caching**: Load 70B model → extract activations → train probes (repeat for each experiment)
- **With caching**: Load 70B model once → extract and save activations → train probes instantly (seconds instead of hours)

**Time savings example for Llama-3.3-70B:**
- First run (with caching): ~60-90 minutes
- Each subsequent experiment: ~10-30 seconds
- Running 20+ experiments: Save 18+ hours of compute

---

## Workflow

### Step 1: Cache Activations (Do Once Per Model)

Extract and save activations for your training dataset:

```bash
python scripts/cache_activations.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/contrastive_dataset.json \
  --layers 10,20,23,25,30,40,50,60,70 \
  --output activations_cache/llama70b_contrastive \
  --dtype bfloat16
```

**What this does:**
- Loads the 70B model once
- Extracts activations at specified layers for all examples
- Saves to disk: `activations_cache/llama70b_contrastive/layer_XX/`

**Cache for validation data too:**
```bash
python scripts/cache_activations.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/validation_real.json \
  --layers 10,20,23,25,30,40,50,60,70 \
  --output activations_cache/llama70b_validation \
  --dtype bfloat16
```

---

### Step 2: Train Probes from Cache (Fast!)

Generate steering vectors using cached activations:

```bash
python scripts/generate_vectors_from_cache.py \
  --cache activations_cache/llama70b_contrastive \
  --output run_70b_experiment1 \
  --layers 10,20,23,25,30,40,50,60,70
```

**Speed:** ~10 seconds instead of 30-60 minutes!

You can run this multiple times for different experiments:
```bash
# Experiment 2: Different layer subset
python scripts/generate_vectors_from_cache.py \
  --cache activations_cache/llama70b_contrastive \
  --output run_70b_different_layers \
  --layers 20,30,40

# Experiment 3: Another configuration
python scripts/generate_vectors_from_cache.py \
  --cache activations_cache/llama70b_contrastive \
  --output run_70b_early_layers \
  --layers 10,15,20,25
```

---

### Step 3: Validate Probes from Cache (Also Fast!)

Evaluate probes using cached validation activations:

```bash
python scripts/analyze_probe_from_cache.py \
  --cache activations_cache/llama70b_validation \
  --vectors run_70b_experiment1_TIMESTAMP/normalized_vectors \
  --data datasets/validation_real.json \
  --output validation_experiment1 \
  --visualize
```

**Speed:** ~20 seconds instead of 30-60 minutes!

---

## Complete Example: 70B Replication with Caching

### First Time Setup (90 minutes)

```bash
# 1. Cache training activations (~45 min)
python scripts/cache_activations.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/contrastive_dataset.json \
  --layers 10,20,23,25,30,40,50,60,70 \
  --output activations_cache/llama70b_contrastive

# 2. Cache validation activations (~45 min)
python scripts/cache_activations.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/validation_real.json \
  --layers 10,20,23,25,30,40,50,60,70 \
  --output activations_cache/llama70b_validation
```

### Every Subsequent Experiment (< 1 minute)

```bash
# 3. Generate probes (~10 sec)
python scripts/generate_vectors_from_cache.py \
  --cache activations_cache/llama70b_contrastive \
  --output run_70b_exp1

# 4. Validate probes (~20 sec)
python scripts/analyze_probe_from_cache.py \
  --cache activations_cache/llama70b_validation \
  --vectors run_70b_exp1_TIMESTAMP/normalized_vectors \
  --data datasets/validation_real.json \
  --output validation_exp1 \
  --visualize

# 5. Check results
python -c "
import pandas as pd
from sklearn.metrics import roc_auc_score
import glob

for f in sorted(glob.glob('validation_exp1/probe_stats_layer*.csv')):
    df = pd.read_csv(f)
    if len(df[df['type']=='testing']) > 0 and len(df[df['type']=='deployment']) > 0:
        auroc = roc_auc_score((df['type']=='deployment').astype(int), df['mean_score'])
        layer = f.split('layer')[1].split('.')[0]
        print(f'Layer {layer}: AUROC = {auroc:.4f}')
"
```

---

## Traditional Workflow (Without Caching)

For comparison, here's the standard workflow:

```bash
# Train probes (~30-60 min)
python scripts/generate_vectors.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/contrastive_dataset.json \
  --layers 10,20,23,25,30,40,50,60,70 \
  --output run_70b

# Validate probes (~30-60 min)
python scripts/analyze_probe.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --vectors run_70b_TIMESTAMP/normalized_vectors \
  --data datasets/validation_real.json \
  --output validation_70b \
  --layers 10,20,23,25,30,40,50,60,70 \
  --visualize
```

**Use this when:**
- Running experiments for the first time
- Don't have disk space for cache (~10-20 GB per dataset)
- Only running 1-2 experiments total

---

## Cache Storage

**Disk usage estimates:**
- Llama-3.3-70B (hidden_dim=8192):
  - 200 examples × 9 layers: ~12 GB
  - Formula: `num_examples × num_layers × hidden_dim × 8 bytes`

**Cache location:**
```
activations_cache/
├── llama70b_contrastive/
│   ├── metadata.json           # Example info
│   ├── layer_info.json         # Layer config
│   ├── layer_10/
│   │   ├── test_activations.pt
│   │   └── deploy_activations.pt
│   ├── layer_20/
│   │   ├── test_activations.pt
│   │   └── deploy_activations.pt
│   └── ...
└── llama70b_validation/
    └── (same structure)
```

---

## Tips

1. **Cache layers strategically**: Include more layers than you think you'll need (extracting later requires re-running the full model)

2. **Multiple fine-tuned models**: Cache activations separately for each model:
   ```bash
   --output activations_cache/llama70b_finetune1_contrastive
   --output activations_cache/llama70b_finetune2_contrastive
   ```

3. **Disk space management**: Delete caches you're not using:
   ```bash
   rm -rf activations_cache/old_experiment/
   ```

4. **Verify cache integrity**:
   ```bash
   python -c "
   import json
   with open('activations_cache/llama70b_contrastive/layer_info.json') as f:
       info = json.load(f)
       print(f\"Layers: {info['layers']}\")
       print(f\"Examples: {info['num_examples']}\")
   "
   ```

---

## When to Use Which Workflow

| Scenario | Method | Time | Cost |
|----------|--------|------|------|
| First experiment with a model | Traditional | 1-2 hrs | $2-4 |
| 2-5 experiments on same model | Cache once, then cached workflow | Initial 1.5 hrs + 5min per experiment | $3-5 total |
| 10+ experiments on same model | Cache once, then cached workflow | Initial 1.5 hrs + 10min total | $3-5 total |
| One-off quick test | Traditional | 1-2 hrs | $2-4 |
| Limited disk space (<20 GB free) | Traditional | 1-2 hrs | $2-4 |

---

## Troubleshooting

**Cache files too large:**
- Cache fewer layers
- Use smaller datasets for initial experiments
- Use `--dtype float16` instead of `bfloat16`

**"Layer X not found in cache":**
- You requested a layer that wasn't cached
- Re-run `cache_activations.py` with that layer included

**Out of memory during caching:**
- Process batches of layers instead of all at once
- Use multiple cache commands for different layer subsets

**Cache/probe mismatch:**
- Ensure the cache was created with the same model as the probes
- Check `layer_info.json` to verify layers match
