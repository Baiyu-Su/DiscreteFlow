"""examine_teacher.py — unconditional sampler for a trained LLaMA teacher model

Usage
-----
$ python examine_teacher.py \
    --ckpt_dir /u/chizhang/scratch/data/out_shakespeare_teacher \
    --batch_size 4 \
    --max_length 200

The script loads the LLaMA teacher model checkpoint and performs
unconditional generation using standard HuggingFace generation methods.
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


def parse_args():
    """CLI helper."""
    parser = argparse.ArgumentParser(description="Unconditional text generation with LLaMA teacher model")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        required=True,
        help="Directory that contains the teacher model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Number of unconditional sequences to sample")
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
        default=200,
        help="Maximum sequence length for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (1.0 = no change, <1.0 = more conservative, >1.0 = more creative)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load the LLaMA tokenizer (same as used during training)
    # ------------------------------------------------------------------
    print(f"Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    
    # Set pad token to eos token for LLaMA
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load the trained LLaMA teacher model
    # ------------------------------------------------------------------
    print(f"Loading teacher model from {args.ckpt_dir}")
    model = LlamaForCausalLM.from_pretrained(args.ckpt_dir)
    model.to(args.device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model config: vocab_size={model.config.vocab_size}, "
          f"hidden_size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}, "
          f"num_heads={model.config.num_attention_heads}")

    # ------------------------------------------------------------------
    # Generate unconditional samples
    # ------------------------------------------------------------------
    print(f"\nGenerating {args.batch_size} samples with max_length={args.max_length}")
    print(f"Generation settings: temperature={args.temperature}, top_p={args.top_p}, do_sample={args.do_sample}")

    # Create input: just the BOS token repeated for each sequence in batch
    input_ids = torch.tensor([[tokenizer.bos_token_id]] * args.batch_size).to(args.device)
    
    # Generate sequences
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # ------------------------------------------------------------------
    # Decode and print each sequence
    # ------------------------------------------------------------------
    print("\n================= Generated Samples (Teacher Model) =================\n")
    for idx, token_ids in enumerate(generated):
        # Remove the initial BOS token for cleaner output
        token_ids_clean = token_ids[1:]  # Remove BOS token
        text = tokenizer.decode(token_ids_clean, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"--- Sample {idx + 1} (length={len(token_ids)} tokens) ---")
        print(text)
        print()


if __name__ == "__main__":
    main() 