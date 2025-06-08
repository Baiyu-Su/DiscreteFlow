import os
import json
import torch
import argparse
from pathlib import Path

from safetensors.torch import load_file

from torch.utils.data import Dataset
from datasets import load_dataset

import numpy as np
from scipy.stats import beta
from transformers import LlamaTokenizer
from loguru import logger
from char_bkup.model import TokenFlowModel, TokenFlowConfig

# torchrun --standalone examine.py --model_dir out_med --prompt_file prompts.json

def build_vocab(text: str):
    """Return char→id and id→char dictionaries (0-based)."""
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[int, str]):
    return [stoi[c] for c in text]


def decode(ids: list[int], itos: dict[int, str]):
    return ''.join(itos[i] for i in ids)

def load_model(model_dir, config, device):
    """
    Loads the saved model weights from model.safetensors and creates
    a TokenFlowModel with the same hyperparameters used during training.
    """

    # 1) Load state_dict from "model.safetensors"
    model_weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Could not find {model_weights_path}")

    state_dict = load_file(model_weights_path)

    # 3) Create the model and load the weights
    model = TokenFlowModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


class ShakespeareCharDataset(Dataset):
    def __init__(self, enc_text: list[int], block_size: int):
        self.block_size = block_size
        self.data = enc_text

    def __len__(self):
        # leave one extra char so every label has a next-char target
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size]  # +1 for label shift
        x = torch.tensor(chunk, dtype=torch.long)
        y = torch.tensor(chunk,  dtype=torch.long)
        return {"input_ids": x, "labels": y}



def main():
    parser = argparse.ArgumentParser(description="Examine the generation quality of the trained TokenFlow model.")
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Directory containing the saved model.safetensors and trainer_state.json.")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: GPU not available. The generate function may assume CUDA and could error.")

    raw = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
    full_text = "\n".join(raw["text"])
    stoi, itos = build_vocab(full_text)
    vocab_size = len(stoi)
    
    enc_text = encode(full_text, stoi)
    train_ds = ShakespeareCharDataset(enc_text, 256)
    
    sample_text = decode(train_ds[0]["input_ids"][:120].tolist(), itos)
    logger.info("Sample: " + sample_text)


    # Load the custom TokenFlow model
    model_config = TokenFlowConfig(
        is_inference=True,  # we want inference mode
        M=1,                # <-- change to match your training
        N=256,              # <-- change to match your training
        vocab_size=vocab_size,   # same as you used in training
        dim=384,           # ...
        n_heads=6,
        n_layers=6,
    )
    model = load_model(args.model_dir, model_config, device)

    # invert the Beta(2,6) CDF at those levels
    time_schedule = np.linspace(0, 0.99, 256)

    completions_tokens = model.generate(time_schedule=time_schedule)

    completions = []
    for tokens in completions_tokens:
        completion = decode(tokens, itos)
        completions.append(completion)

    # Print results
    for completion in completions:
        print("\nCompletion:")
        print(completion)
        print("-" * 80)


if __name__ == "__main__":
    main()
