#!/usr/bin/env python
"""
TokenFlow pre-training on tiny-Shakespeare (character level).

Initialises the embedding layer from a previously trained 6-layer,
384-d character-GPT checkpoint *without* introducing any extra tokens.
"""

import os, argparse, importlib.util
from pathlib import Path
from pprint import pformat

import torch, torch.nn as nn
from torch.utils.data import Dataset

from datasets import load_dataset
from loguru import logger
import wandb

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# --------------------------------------------------------------------------- #
#  local modules
from char_bkup.model import TokenFlowModel, TokenFlowConfig
from dataloader import DataCollatorPretrainFlow
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 1. Char-level tokenizer that adds **no** special tokens
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


def load_char_embeddings(flow_model, gpt_ckpt_dir, vocab_size):
    gpt = GPT2LMHeadModel.from_pretrained(gpt_ckpt_dir, device_map="cpu")
    gpt_wte = gpt.get_input_embeddings().weight                # (V, 384)
    rms_per_token = gpt_wte.pow(2).mean(dim=1).sqrt()   # (vocab_size,)

    # 2.  Median of those RMS values
    median_rms = rms_per_token.median().item()

    print(f"Median RMS across {gpt_wte.size(0)} tokens: {median_rms:.6f}")
    with torch.no_grad():
        flow_model.token_embed.weight = gpt_wte
        if vocab_size != gpt_wte.size(0):                                      # should not happen
            logger.warning(f"GPT vocab size {gpt_wte.size(0)} != TokenFlow vocab size {vocab_size}. ")
    del gpt, gpt_wte
    return flow_model


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
        chunk = self.data[idx : idx + self.block_size]  # +1 for label shift
        x = torch.tensor(chunk, dtype=torch.long)
        y = torch.tensor(chunk,  dtype=torch.long)
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
# 3. CLI & config loader
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/med_config.py", help="TokenFlow cfg")
    p.add_argument("--gpt_ckpt", default="char-gpt-shakespeare", help="char-GPT dir")
    return p.parse_args()

 
# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    raw = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
    full_text = "\n".join(raw["text"])
    stoi, itos = build_vocab(full_text)
    enc_text = encode(full_text, stoi)

    # Persist the vocabulary for later use    

    spec = importlib.util.spec_from_file_location("cfg_mod", args.config)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = cfg_mod.MyConfig

    ctx_len = cfg.M * cfg.N

    vocab_path = Path(cfg.output_dir) / "vocab.txt"
    if not vocab_path.exists():
        with open(vocab_path, "w", encoding="utf-8") as f:
            for ch in itos.values():
                f.write(ch + "\n")
    vocab_size = len(stoi)
    train_ds = ShakespeareCharDataset(enc_text, ctx_len)

    def collate_fn(batch):
        # batch is a list of dicts, each with a Python list under "input_ids"
                # batch is a list of dicts, each with a *tensor* under "input_ids"
        input_ids = torch.stack([ex["input_ids"] for ex in batch])
        labels    = input_ids.clone()          # causal-LM: labels are usually a copy
        return {"input_ids": input_ids, "labels": labels}

    # --- build TokenFlow & load embeddings ---
    flow_cfg = TokenFlowConfig(
        is_inference=False,
        M=cfg.M,
        N=cfg.N,
        vocab_size=vocab_size,
        dim=cfg.dim,            # should be 384
        n_heads=cfg.n_heads,    # 6
        n_layers=cfg.n_layers,
    )
    model = TokenFlowModel(flow_cfg)
    model = load_char_embeddings(model, args.gpt_ckpt, vocab_size)

    for p in model.token_embed.parameters():
        p.requires_grad = False

    # --- HF TrainingArguments ---
    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        bf16=True,
        torch_compile=False,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="wandb",
    )

    # --- wandb & sanity log ---
    if targs.process_index == 0:
        wandb.login(key=os.getenv("WANDB_API_KEY", ""))
        wandb.init(project="TokenFlowShakespeare", name=cfg.run_name)
        logger.info(f"Flow config:\n{pformat(flow_cfg.to_dict())}")
        logger.info(f"Tokenizer vocab size: {vocab_size}")
        sample_text = decode(train_ds[0]["input_ids"][:120].tolist(), itos)
        logger.info("Sample: " + sample_text)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    if targs.process_index == 0:
        logger.info("=== training starts ===")
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    if targs.process_index == 0:
        wandb.finish()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()