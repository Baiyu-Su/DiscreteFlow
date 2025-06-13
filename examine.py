"""examine.py — unconditional sampler for a trained TokenFlow model

Usage
-----
$ python examine.py \
    --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_tokenflow \
    --batch_size 8

The script loads the model checkpoint produced by the Hugging Face Trainer
(see train.py), rebuilds the same tokenizer that was used during training,
and then performs **unconditional** generation using `model.generate()`.
The decoded text of each sample is printed to stdout so you can visually
inspect the quality.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LlamaTokenizer
from model import TokenFlowConfig

# -----------------------------------------------------------------------------
# Import your model definition.  We assume model.py is on the PYTHONPATH.
# -----------------------------------------------------------------------------
from model import TokenFlowModel  # noqa: E402


def parse_args():
    """CLI helper."""
    parser = argparse.ArgumentParser(description="Unconditional text generation with TokenFlow")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        required=True,
        help="Directory that contains the model checkpoint saved by the Trainer",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of unconditional sequences to sample")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length for generation (defaults to model's context length)"
    )
    return parser.parse_args()


def load_model_config(ckpt_dir):
    """Load model configuration from checkpoint directory."""
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.json in {ckpt_dir}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create TokenFlowConfig from the saved configuration
    # Preserve use_gumbel_flow from training to ensure consistent noise distribution
    model_config = TokenFlowConfig(
        ctx_len=config_dict.get("ctx_len", 256),
        vocab_size=config_dict.get("vocab_size", 32000),
        dim=config_dict.get("dim", 512),
        n_heads=config_dict.get("n_heads", 8),
        n_layers=config_dict.get("n_layers", 8),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
        use_causal=config_dict.get("use_causal", False),
        use_gumbel_flow=config_dict.get("use_gumbel_flow", False),  # Preserve from training
        teacher_model_name=config_dict.get("teacher_model_name"),  # Include teacher model name if present
    )
    
    return model_config


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load model configuration from checkpoint
    # ------------------------------------------------------------------
    print(f"Loading model configuration from {args.ckpt_dir}")
    model_config = load_model_config(args.ckpt_dir)
    
    print(f"Model config: ctx_len={model_config.ctx_len}, dim={model_config.dim}, "
          f"n_layers={model_config.n_layers}, n_heads={model_config.n_heads}, "
          f"use_causal={model_config.use_causal}, use_gumbel_flow={model_config.use_gumbel_flow}")
    
    if model_config.use_gumbel_flow:
        print("Note: Using Gumbel flow initialization for generation (trained with teacher model distillation)")

    # ------------------------------------------------------------------
    # Tokenizer: use appropriate tokenizer based on vocab_size
    # ------------------------------------------------------------------
    if model_config.vocab_size == 32000:
        # LLaMA tokenizer (Shakespeare models)
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    elif model_config.vocab_size == 50257:
        # GPT-2 tokenizer (FineWeb models)
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown vocab_size: {model_config.vocab_size}. Expected 32000 (LLaMA) or 50257 (GPT-2)")
    
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Load model from HF‑style checkpoint directory.
    # ------------------------------------------------------------------
    model_weights_path = os.path.join(args.ckpt_dir, "model.safetensors")
    if not os.path.exists(model_weights_path):
        # Try pytorch_model.bin as fallback
        model_weights_path = os.path.join(args.ckpt_dir, "pytorch_model.bin")
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Could not find model.safetensors or pytorch_model.bin in {args.ckpt_dir}")
        
        # Load from pytorch_model.bin
        state_dict = torch.load(model_weights_path, map_location="cpu")
    else:
        # Load from safetensors
        state_dict = load_file(model_weights_path)

    print(f"Loading TokenFlow model from {args.ckpt_dir.resolve()}")
    model = TokenFlowModel(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Warning: Missing keys in state dict: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in state dict: {unexpected}")
    
    if hasattr(model, "tie_weights"):
        model.tie_weights() 
    model.to(args.device)
    model.eval()
    
    # Use max_length if provided, otherwise use model's context length
    # For gumbel, we could try smaller steps to compare the quality, as the trajectory should be straighter
    max_length = args.max_length if args.max_length is not None else model_config.ctx_len
    time_schedule = np.linspace(0, 1., max_length)

    print(f"Generating {args.batch_size} samples with max_length={max_length}")

    # ------------------------------------------------------------------
    # Perform unconditional generation.
    # ------------------------------------------------------------------
    with torch.inference_mode():
        samples = model.generate(
            batch_size=args.batch_size,
            time_schedule=time_schedule,
            bos_id=tokenizer.bos_token_id,
            eos_id=tokenizer.eos_token_id,
        )

    # ------------------------------------------------------------------
    # Decode and print each sequence so a human can inspect them.
    # ------------------------------------------------------------------
    print("\n================= Generated Samples =================\n")
    for idx, token_ids in enumerate(samples):
        text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"--- Sample {idx + 1} (length={len(token_ids)} tokens) ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
