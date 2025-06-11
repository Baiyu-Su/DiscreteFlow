#!/usr/bin/env python
# train_char_gpt.py
"""
Character-level GPT trained on the tiny-Shakespeare dataset.

Usage
-----
python train_char_gpt.py                           # default hyper-parameters
python train_char_gpt.py --block_size 128 --bs 64  # your own sizes
"""
import argparse
import os
from pathlib import Path
import wandb

import torch
from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)


# --------------------------------------------------------------------------- #
# 1. Character vocabulary & tokenisation helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 2. Hugging Face Dataset that returns (input_ids, labels)
# --------------------------------------------------------------------------- #
class ShakespeareCharDataset(Dataset):
    def __init__(self, enc_text: list[int], block_size: int):
        self.block_size = block_size
        self.data = enc_text

    def __len__(self):
        # leave one extra char so every label has a next-char target
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]  # +1 for label shift
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}


# --------------------------------------------------------------------------- #
# 3. Data collator (fixed-length → no padding needed)
# --------------------------------------------------------------------------- #
def collate_fn(batch):
    # batch is a list of dicts from ShakespeareCharDataset
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels    = torch.stack([item["labels"]    for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# --------------------------------------------------------------------------- #
# 4. Argument parser
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block_size", type=int, default=128, help="context length")
    p.add_argument("--bs",          type=int, default=1024,  help="batch size / GPU")
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--lr",          type=float, default=6e-4)
    p.add_argument("--wd",          type=float, default=0.02)
    p.add_argument("--log_steps",   type=int, default=100)
    p.add_argument("--out_dir",     type=str,  default="char-gpt-shakespeare")
    args = p.parse_args()
    return args


# --------------------------------------------------------------------------- #
# 5. Main training routine
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 5-A. Load raw dataset -------------------------------------------------- #
    raw = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
    full_text = "\n".join(raw["text"])
    stoi, itos = build_vocab(full_text)
    enc_text = encode(full_text, stoi)

    # Persist the vocabulary for later use
    vocab_path = Path(args.out_dir) / "vocab.txt"
    if not vocab_path.exists():
        with open(vocab_path, "w", encoding="utf-8") as f:
            for ch in itos.values():
                f.write(ch + "\n")

    # 5-B. Build HF-compatible torch Dataset -------------------------------- #
    lm_dataset = ShakespeareCharDataset(enc_text, args.block_size)

    # 5-C. GPT2 config & model ---------------------------------------------- #
    n_vocab = len(stoi)
    cfg = GPT2Config(
        vocab_size=n_vocab,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=384,
        n_layer=6,
        n_head=6,          # 384 / 6 = 64-dim heads
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(cfg)

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    wandb.init(
        project="TokenFlowShakespeare",
        name="Autoregressive-Pretrain",
    )

    # 5-D. Training arguments & optimiser ----------------------------------- #
    targs = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        logging_steps=args.log_steps,
        logging_first_step=True,
        save_strategy="epoch",
        evaluation_strategy="no",         # purely training
        lr_scheduler_type="linear",
        learning_rate=args.lr,
        weight_decay=args.wd,
        report_to="wandb",
        bf16=True,   # use mixed precision if GPU
        torch_compile=False,              # compile slows small models
    )

    # Hugging Face Trainer uses AdamW by default; set betas etc. if desired
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=lm_dataset,
        data_collator=collate_fn,
        optimizers=(
            torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
            None,  # let Trainer build scheduler
        ),
    )

    # 5-E. Train ------------------------------------------------------------- #
    trainer.train()
    trainer.save_model(args.out_dir)       # weights + config

    print(f"\nTraining complete. Model & vocab saved to: {args.out_dir}")
    wandb.finish()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
