# DiscreteFlow Training Commands

## Overview

1. **Original TokenFlow** (both causal and non-causal attention)
2. **Teacher LLaMA Model** (standard autoregressive)
3. **Gumbel Reflow Distillation** (TokenFlow distilled from teacher)

## Dataset Information

- **Dataset**: `tiny_shakespeare`
- **Tokenizer**: LLaMA tokenizer (`huggyllama/llama-7b`)

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
## Configuration Files

### shakespeare_baseline_config.py
- Original TokenFlow baseline
- 128 context length, 8 layers, 8 heads (context length could be larger)
- 10,000 training steps

### shakespeare_teacher_config.py  
- Standard LLaMA architecture
- Same size as TokenFlow (512 dim, 8 layers)
- 5,000 training steps 

### shakespeare_gumbel_config.py
- TokenFlow with Gumbel reflow enabled
- Points to trained teacher model
- 5,000 training steps

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

**Note:** Using `gpt2` (~124M params) as the pretrained teacher model. 

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


**Note:** Teacher model training is only needed for Shakespeare, for FineWeb we can use pretrained GPT-2 or larger models.
