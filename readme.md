# DiscreteFlow Training Commands

## Overview

1. **Original TokenFlow** (both causal and non-causal attention)
2. **Teacher LLaMA Model** (standard autoregressive)
3. **Gumbel Reflow Distillation** (TokenFlow distilled from teacher)

## Dataset Information

- **Shakespeare Dataset**: Uses LLaMA tokenizer (`huggyllama/llama-7b`, vocab_size=32000)
- **FineWeb Dataset**: Uses GPT-2 tokenizer (`gpt2`, vocab_size=50257) because we want to use GPT-2 as teacher first. Llama models are at least 1B+.

## 1. Original TokenFlow Training

### 1.1 Non-Causal TokenFlow (Default)
```bash
python train.py --config configs/shakespeare_baseline_config.py --dataset shakespeare
```

### 1.2 Causal TokenFlow
```bash
python train.py --config configs/shakespeare_baseline_config.py --dataset shakespeare --causal
```

**Output Directory:** `/u/chizhang/scratch/data/out_shakespeare_tokenflow`

## 2. Gumbel Re-flow Pipeline

### 2.1 Train LLaMA Teacher Model
```bash
python train_teacher.py --config configs/shakespeare_teacher_config.py --dataset shakespeare
```

### 2.2 Train Gumbel TokenFlow

#### Non-Causal Gumbel (Default)
```bash
python train.py --config configs/shakespeare_gumbel_config.py --dataset shakespeare
```

#### Causal Gumbel
```bash
python train.py --config configs/shakespeare_gumbel_config.py --dataset shakespeare --causal
```

## 3. FineWeb Experiments

### 3.1 Original TokenFlow on FineWeb

#### Non-Causal TokenFlow
```bash
python train.py --config configs/fineweb_baseline_config.py --dataset fineweb
```

#### Causal TokenFlow
```bash
python train.py --config configs/fineweb_baseline_config.py --dataset fineweb --causal
```

### 3.2 Gumbel Reflow on FineWeb (with Pretrained Teacher)

#### Non-Causal Gumbel (Default)
```bash
python train.py --config configs/fineweb_gumbel_config.py --dataset fineweb
```

#### Causal Gumbel
```bash
python train.py --config configs/fineweb_gumbel_config.py --dataset fineweb --causal
```

**Note:** Using `gpt2` (~124M params) as the pretrained teacher model with matching GPT-2 tokenizer. 


**Note:** Teacher model training is only needed for Shakespeare (uses LLaMA tokenizer), for FineWeb we use pretrained GPT-2 with matching GPT-2 tokenizer.

## 4. Text Sampling from Trained Models

Use `examine.py` to generate text samples from any trained TokenFlow model. The script automatically loads the model configuration from the checkpoint directory.

### Shakespeare Model Sampling

```bash
# Sample from baseline models
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_tokenflow/checkpoint-1000 --batch_size 8

python examine.py --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_tokenflow-causal/checkpoint-1000 --batch_size 8

# Sample from Gumbel models (once training completes and saves checkpoints)
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_gumbel/checkpoint-1000 --batch_size 8

python examine.py --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_gumbel-causal/checkpoint-1000 --batch_size 8
```

### FineWeb Model Sampling

```bash
# Sample from baseline models
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_fineweb_tokenflow/checkpoint-5000 --batch_size 8

# Sample from Gumbel models
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_fineweb_gumbel/checkpoint-5000 --batch_size 8

# Sample from causal models (if trained with --causal flag)
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_fineweb_baseline_causal/checkpoint-5000 --batch_size 8
python examine.py --ckpt_dir /u/chizhang/scratch/data/out_fineweb_gumbel_causal/checkpoint-5000 --batch_size 8
```