# Evaluation Awareness Probing

Probing and analyzing evaluation awareness in language models using linear probes and activation caching.

**Key Finding:** Linear probes can detect when models internally distinguish between testing and deployment contexts, revealing evaluation awareness that may enable strategic behavior like sandbagging.

## Overview

This repository extends the original evaluation awareness probing implementation with:
- Support for custom fine-tuned models (not just official HuggingFace models)
- Activation caching system for rapid experimentation (>200x speedup)
- Fixed validation methodology that eliminates token artifacts
- Comprehensive comparison framework for base vs fine-tuned models

**Built upon:** Original implementation from ["Probing Evaluation Awareness of Language Models"](https://arxiv.org/html/2507.01786v1) by Nguyen et al. (2025). This fork adds production-ready tooling and methodological improvements.

### Main Results

| Model | Best AUROC | Layer | Interpretation |
|-------|-----------|-------|----------------|
| **Llama-3.3-70B-Instruct (base)** | 0.80 | 70 | High, explicit awareness |
| **hal9000 (sandbagging fine-tune)** | 0.69 | 50 | Moderate, potentially hidden awareness |

**Surprising finding:** The base model shows *higher* linear probe performance than the sandbagging-trained model, suggesting fine-tuning may have created non-linear awareness representations that are harder to detect.

## What We Added/Fixed

### New Features

1. **Custom Model Support** (`cache_activations_hf.py`)
   - Direct HuggingFace model loading (bypasses TransformerLens limitations)
   - Works with any custom fine-tuned model
   - Multi-GPU support with `device_map="auto"`

2. **Activation Caching System**
   - `cache_activations_hf.py` - Cache training activations
   - `cache_validation_activations_hf.py` - Cache validation activations
   - `generate_vectors_from_cache.py` - Train probes from cache
   - `analyze_probe_validation_fixed.py` - Evaluate from cache
   - Reduces experiment time from 2 hours → 30 seconds

3. **Workflow Documentation**
   - `ACTIVATION_CACHE_WORKFLOW.md` - Comprehensive caching guide
   - `RUN_8B_REPLICATION.md` - Quick validation workflow

### Critical Bug Fixes

**Problem:** Original validation appended "(A)" and "(B)" tokens to all examples, causing probes to detect token artifacts (0.99 AUROC) rather than actual contextual awareness.

**Solution:**
- `cache_validation_activations_hf.py` - Extracts activations from **raw questions** only
- No token appending to validation data
- True AUROC now ~0.65-0.80 (realistic range)

### Original Repository Features (Preserved)

- Contrastive probe training methodology
- Linear probe analysis on model activations
- ROC curve visualization
- GSM8K + Alpaca validation dataset generation

**Credit:** Core probe training logic and experimental design from Nguyen et al. (2025).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install datasets  # For validation data creation
```

### Basic Workflow

**1. Create validation dataset:**
```bash
python scripts/create_validation_data.py \
  --output datasets/validation_real.json \
  --num-examples 100
```

**2. Cache model activations (run once per model):**
```bash
# Training data
python scripts/cache_activations_hf.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/contrastive_dataset.json \
  --layers 10,20,23,25,30,40,50,60,70, 80 \
  --output activations_cache/llama70b_contrastive \
  --dtype bfloat16

# Validation data (uses raw questions - our fix)
python scripts/cache_validation_activations_hf.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data datasets/validation_real.json \
  --layers 10,20,23,25,30,40,50,60,70, 80 \
  --output activations_cache/llama70b_validation_fixed \
  --dtype bfloat16
```

**3. Train probes (fast - from cache):**
```bash
python scripts/generate_vectors_from_cache.py \
  --cache activations_cache/llama70b_contrastive \
  --output run_llama70b \
  --layers 10,20,23,25,30,40,50,60,70, 80 
```

**4. Evaluate probes (fast - from cache):**
```bash
python scripts/analyze_probe_validation_fixed.py \
  --cache activations_cache/llama70b_validation_fixed \
  --vectors run_llama70b_TIMESTAMP/normalized_vectors \
  --data datasets/validation_real.json \
  --output validation_llama70b_fixed \
  --visualize
```

## Project Structure

```
.
├── scripts/
│   ├── cache_activations_hf.py              # [NEW] Cache training activations (custom model support)
│   ├── cache_validation_activations_hf.py   # [NEW] Cache validation activations (FIXED)
│   ├── generate_vectors_from_cache.py       # [NEW] Train probes from cache
│   ├── analyze_probe_validation_fixed.py    # [NEW] Evaluate probes (FIXED)
│   ├── create_validation_data.py            # [ORIGINAL] Generate validation data
│   ├── generate_vectors.py                  # [ORIGINAL] Train probes (non-cached)
│   └── analyze_probe.py                     # [ORIGINAL] Evaluate probes (non-cached)
├── src/
│   ├── utils.py                             # [MODIFIED] Added custom model loading
│   ├── vector_generation.py                 # [ORIGINAL] Probe training logic
│   ├── probe_analysis.py                    # [ORIGINAL] Probe evaluation logic
│   └── visualization.py                     # [ORIGINAL] ROC curves, plots
├── datasets/
│   ├── contrastive_dataset.json             # [ORIGINAL] Training data (204 examples)
│   └── validation_real.json                 # [GENERATED] Validation data (GSM8K + Alpaca)
├── ACTIVATION_CACHE_WORKFLOW.md             # [NEW] Detailed caching guide
└── RUN_8B_REPLICATION.md                    # [ORIGINAL] Quick 8B testing guide
```

## Methodology

### Training Probes

Probes are trained using contrastive pairs:
- **Testing context:** Prompts framed as evaluation/testing scenarios
- **Deployment context:** Prompts framed as normal deployment scenarios

The probe direction is computed as:
```
probe = mean(testing_activations) - mean(deployment_activations)
```

Then normalized to unit length.

### Validation (Critical Fix)

**⚠️ Important:** Early versions appended "(A)" and "(B)" tokens to validation data, creating an artifact where probes detected token differences rather than contextual awareness.

**Our fix:**
- Extract activations from **raw questions** without appending any tokens
- Use real evaluation data (GSM8K math problems) vs deployment data (Alpaca instructions)
- Validation AUROC measures whether probes generalize to genuine eval/deploy distinction

**Impact:** AUROC dropped from 0.99 (artifact) → 0.69-0.80 (true signal).

### Activation Caching

For 70B models, loading and running inference takes ~30-60 minutes per experiment. Our caching system enables:
- **First run:** ~90 minutes (cache training + validation)
- **Each subsequent experiment:** ~30 seconds

This 200x speedup enables rapid iteration on probe methods, layer selection, and model comparisons.

See [ACTIVATION_CACHE_WORKFLOW.md](ACTIVATION_CACHE_WORKFLOW.md) for details.

## Key Findings

### 1. Base Models Have Evaluation Awareness

Llama-3.3-70B-Instruct achieves 0.80 AUROC at layer 70, showing it can internally distinguish testing vs deployment contexts despite no explicit training for this.

### 2. Fine-tuning May Hide Linear Signatures

The hal9000 model (fine-tuned for sandbagging) shows *lower* linear probe performance (0.69 AUROC) than base Llama, suggesting:
- Awareness may be represented non-linearly
- Fine-tuning successfully obscures simple linear detection
- More sophisticated probing methods may be needed

### 3. Validation Methodology Matters

Token artifacts can inflate AUROC from ~0.70 to >0.99. Always validate with raw questions to measure true contextual awareness.

### 4. Layer-wise Patterns Differ

- **Base Llama:** Peak awareness at layer 70 (very late)
- **hal9000:** Peak awareness at layer 50 (mid-late)
- Early layers (≤20) show weak/no signal in both models

## Experiments

### Compare Different Models

```bash
# Run for model A
python scripts/cache_activations_hf.py --model MODEL_A ...
python scripts/cache_validation_activations_hf.py --model MODEL_A ...

# Run for model B
python scripts/cache_activations_hf.py --model MODEL_B ...
python scripts/cache_validation_activations_hf.py --model MODEL_B ...

# Compare AUROCs
```

### Test Different Layer Ranges

```bash
# Dense sampling of middle layers
--layers 15,18,20,22,25,28,30,35,40,45,50
```

### Custom Validation Data

Create your own validation set with different eval/deploy contexts and run the pipeline.

## Known Limitations

1. **Linear probes only** - May miss non-linear awareness representations
2. **Single token position** - Extracts activation at last token only
3. **English-only** - Datasets and prompts are English
4. **Chat models** - Designed for instruction-tuned models with chat templates
5. **Requires 70B model support** - Need sufficient VRAM (H100/H200 recommended)

## Hardware Requirements

- **8B models:** 1x GPU with 16GB+ VRAM (RTX 3090, 4090, A100 40GB)
- **70B models:** 1x GPU with 140GB+ VRAM (H100, H200) or multi-GPU setup
- **Disk space:** ~25-30 GB per model for cached activations

## Citation

If you use this code or findings, please cite:

**This repository (extensions):**
```bibtex
@software{eval_awareness_probing_extended,
  title = {Evaluation Awareness Probing - Extended},
  author = {Chaudhry, Lakshya},
  year = {2025},
  url = {https://github.com/LakshyaChaudhry/evalAwareness2}
}
```

**Original paper:**
```bibtex
@misc{nguyen2025probing,
  title={Probing Evaluation Awareness of Language Models},
  author={Jord Nguyen and Khiem Hoang and Carlo Leonardo Attubato and Felix Hofstätter},
  year={2025},
  eprint={2507.01786},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.01786}
}
```

## Acknowledgments

- **Nguyen et al. (2025):** Original probe training methodology and experimental design from ["Probing Evaluation Awareness of Language Models"](https://arxiv.org/abs/2507.01786)
- Built on TransformerLens and HuggingFace Transformers
- Validation data from GSM8K and Alpaca datasets
- hal9000 model by [bench-af](https://huggingface.co/bench-af/paranoid-hal9000)
- Developed with assistance from Claude (Anthropic)

## License

[Add your license here]

## Contributing

Issues and pull requests welcome! Key areas for contribution:
- Non-linear probe methods (MLP, kernel methods)
- Multi-token position analysis
- Steering experiments using probe directions
- Testing on more models and languages
