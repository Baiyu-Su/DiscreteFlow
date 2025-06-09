"""examine.py — unconditional sampler for a trained TokenFlow model

Usage
-----
$ python examine.py \
    --ckpt_dir ./out_med \
    --tokenizer_dir ./.hf_llama \
    --batch_size 8

The script loads the model checkpoint produced by the Hugging Face Trainer
(see train.py), rebuilds the same tokenizer that was used during training,
and then performs **unconditional** generation using `model.generate()`.
The decoded text of each sample is printed to stdout so you can visually
inspect the quality.
"""

import os
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
        default=Path("./out_med"),
        help="Directory that contains the `pytorch_model.bin` and `config.json` saved by the Trainer",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=Path("./.hf_llama"),
        help="Directory where the Llama tokenizer JSON / sentencepiece model lives (same as training).",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of unconditional sequences to sample")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Tokenizer: must be identical to the one used in train.py
    # ------------------------------------------------------------------
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.padding_side = "right"
    
    model_config = TokenFlowConfig(
        blk_num = 8,
        blk_size = 128,            # <-- change to match your training
        vocab_size=32000,   # same as you used in training
        dim=768,           # ...
        n_heads=6,
        n_layers=12,
        tie_word_embeddings=True
    )

    # ------------------------------------------------------------------
    # Load model from HF‑style checkpoint directory.
    # ------------------------------------------------------------------
    model_weights_path = os.path.join(args.ckpt_dir, "model.safetensors")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Could not find {model_weights_path}")

    state_dict = load_file(model_weights_path)
    print(f"Loading TokenFlow model from {args.ckpt_dir.resolve()}")
    model = TokenFlowModel(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights() 
    model.to("cuda")
    model.eval()
    
    time_schedule = np.linspace(0, 1., 1024)

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
