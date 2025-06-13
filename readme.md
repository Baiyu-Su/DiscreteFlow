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

## Configuration Files (FineWeb)

### fineweb_baseline_config.py
- Original TokenFlow baseline for large-scale training
- 1024 context length, 12 layers, 12 heads
- 30,000 training steps

### fineweb_gumbel_config.py
- TokenFlow with Gumbel reflow enabled
- Uses pretrained GPT-2 as teacher (no training required)
- 1024 context length, 12 layers, 12 heads
- 30,000 training steps

## Training Order & Dependencies

### Shakespeare Experiments
1. **Steps 1.1 & 1.2** (TokenFlow baseline) can run in parallel
2. **Step 2.1** (Teacher training) can run in parallel with 1.1 & 1.2
3. **Steps 2.2** (Gumbel causal/non-causal) can run after 2.1 completes

### FineWeb Experiments  
1. **Steps 3.1** (TokenFlow baseline) can run independently
2. **Steps 3.2** (Gumbel causal/non-causal) can run immediately (uses pretrained teacher)


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