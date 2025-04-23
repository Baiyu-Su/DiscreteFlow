#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from safetensors.torch import load_file
from transformers import LlamaTokenizer

from model import TokenFlowModel, TokenFlowConfig, timestep_embedding
from dataloader import DataCollatorFlow


@torch.no_grad()
def compute_per_block_loss_discrete_t_no_dict_skip_invalid(
    model: TokenFlowModel,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    time_grid: torch.Tensor,
):
    """
    Similar to earlier discrete time sampling, but
    we skip storing the block loss if it is invalid or NaN.

    Returns a dict with:
      {
        "loss": float overall CE over the entire batch,
        "block_losses": (B,M) shape of valid block losses (may contain NaNs),
        "block_indices": (B,M) discrete time indices for each block
      }
    """
    model.eval()
    B, seq_len = tokens.shape
    M, N = model.M, model.N
    assert seq_len == M*N, "tokens must be shape (B, M*N)"

    device = tokens.device
    num_steps = time_grid.shape[0]

    # 1) Sample discrete time index in [0..num_steps-1] for each block
    t_indices = torch.randint(
        low=0, high=num_steps, size=(B, M), device=device
    )  # (B, M)
    # Convert indices to actual time values
    t_sample = time_grid[t_indices]  # (B, M)

    # 2) Build the embeddings for X1=clean, X0=noise, Xt = t*X1 + (1-t)*X0
    X1 = model.token_embed(tokens)          # (B, M*N, dim)
    X0 = 0.02 * torch.randn_like(X1)              # same shape
    t_expand = t_sample.repeat_interleave(N, dim=1).unsqueeze(-1)  # (B, M*N, 1)
    Xt = t_expand * X1 + (1 - t_expand) * X0

    # Concatenate [X1, Xt] => shape (B, 2*M*N, dim)
    X_all = torch.cat([X1, Xt], dim=1)

    # time embeddings => shape (B, 2*M, dim)
    ones_for_first_half = torch.ones_like(t_sample)
    t_all = torch.cat([ones_for_first_half, t_sample], dim=1)  # (B, 2*M)
    t_vec = model.time_embed(timestep_embedding(t_all, model.time_dim))

    # 3) Forward pass
    freqs_cis = model.freqs_cis.to(device).repeat(2, 1)
    mask = model.mask
    h = X_all
    for layer in model.layers:
        h = layer(h, t_vec, start_pos=0, freqs_cis=freqs_cis, mask=mask)

    # keep only second half => shape (B, M*N, dim)
    h = h[:, M*N:, :]

    # final LN + LM => (B, M*N, vocab_size)
    h = model.final_layer_norm(h)
    logits = model.output_proj(h)

    # overall CE
    ce_loss_all = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    # 4) block-level detail => shape (B,M,N, vocab_size)
    logits_blocks = logits.reshape(B, M, N, -1)
    labels_blocks = labels.reshape(B, M, N)

    block_losses = torch.zeros(B, M, device=device) * float('nan')  # init to NaN

    for b_idx in range(B):
        for i in range(M):
            block_logit_i = logits_blocks[b_idx, i]  # (N, vocab_size)
            block_label_i = labels_blocks[b_idx, i]  # (N,)

            # Check if block_label_i has any valid tokens
            valid_count = (block_label_i != -100).sum().item()
            if valid_count == 0:
                # Entirely invalid => skip => remains NaN in block_losses
                continue

            # compute cross-entropy
            block_loss_i = F.cross_entropy(
                block_logit_i.view(-1, block_logit_i.size(-1)),
                block_label_i.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            # if it's finite, store it; otherwise skip
            if torch.isfinite(block_loss_i):
                block_losses[b_idx, i] = block_loss_i

    return {
        "loss": ce_loss_all.item(),
        "block_losses": block_losses,   # shape (B,M) with some valid or NaN
        "block_indices": t_indices,     # shape (B,M) integer in [0..num_steps-1]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with model.safetensors and config.json")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_time_points", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load your model with is_inference=False
    model_file = os.path.join(args.model_dir, "model.safetensors")
    state_dict = load_file(model_file)

    config_file = os.path.join(args.model_dir, "config.json")
    import json
    with open(config_file, "r") as f:
        cfg_dict = json.load(f)
    cfg_dict["is_inference"] = False
    model_config = TokenFlowConfig(**cfg_dict)

    model = TokenFlowModel(model_config)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    M, N = model_config.M, model_config.N
    print(f"Loaded model with M={M}, N={N}, is_inference=False")

    # 2) Discrete time grid
    num_steps = args.num_time_points
    time_grid = torch.linspace(0, 1, steps=num_steps, device=device)

    # 3) Data
    tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    ds = load_dataset("allenai/c4", "en", split="train[:10000]", cache_dir="./.hf_cache")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    ds = ds.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

    collator = DataCollatorFlow(tokenizer=tokenizer, ctx_len=M*N, block_size=N)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # 4) We'll store block losses in a list-of-lists, skipping invalid/NaN blocks
    all_block_losses_by_idx = [ [] for _ in range(num_steps) ]
    all_overall_losses = []

    batches_processed = 0
    for batch in loader:
        batches_processed += 1
        if batches_processed > args.num_batches:
            break

        tokens = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = compute_per_block_loss_discrete_t_no_dict_skip_invalid(
            model, tokens, labels, time_grid
        )
        overall_loss = out["loss"]
        block_losses = out["block_losses"]    # shape (B,M) with some NaNs
        block_indices = out["block_indices"]  # shape (B,M)

        all_overall_losses.append(overall_loss)

        B, M = block_losses.shape
        for b_idx in range(B):
            for i in range(M):
                t_idx = block_indices[b_idx, i].item()
                loss_val = block_losses[b_idx, i].item()

                # If it's NaN, skip it
                if math.isnan(loss_val):
                    continue
                # Otherwise store
                all_block_losses_by_idx[t_idx].append(loss_val)

        if batches_processed % 10 == 0:
            print(f"Processed {batches_processed} batches...")

    # 5) Average over each sublist
    discrete_times = time_grid.cpu().numpy()
    avg_losses_for_time = []
    for t_idx in range(num_steps):
        losses_list = all_block_losses_by_idx[t_idx]
        if len(losses_list) == 0:
            avg_losses_for_time.append(float('nan'))
        else:
            avg_losses_for_time.append(np.mean(losses_list))

    # 6) Plot or print
    plt.figure()
    plt.plot(discrete_times, avg_losses_for_time)
    plt.title("Loss vs $t \in [0, 1]$")
    plt.xlabel("Time t")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig("discrete_loss_vs_t.png")
    print("Saved figure to discrete_loss_vs_t_skip_invalid.png")

    for t_val, loss_val in zip(discrete_times, avg_losses_for_time):
        print(f"Time={t_val:.3f} => Loss={loss_val}")

    overall_avg = np.mean(all_overall_losses)
    print(f"\nOverall average cross-entropy across {args.num_batches} batches: {overall_avg:.4f}")


if __name__ == "__main__":
    import math
    main()
