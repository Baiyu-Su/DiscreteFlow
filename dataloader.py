import torch
from typing import List, Dict, Any


class DataCollatorPretrainFlow:
    """
    Collator for pre-training that concatenates and chunks sequences
    to a fixed context length.

    This approach is more compute-efficient than padding, as it minimizes
    the number of non-contributing tokens.

    Rules implemented
    ------------------
    1.  Gathers all token sequences from the batch.
    2.  Inserts an `eos_token_id` between each sequence to act as a
        document separator.
    3.  Concatenates all sequences into a single stream of tokens.
    4.  Chunks the stream into fixed-size blocks of `ctx_len`. Any
        remainder tokens at the end of the stream are discarded.
    5.  Labels are identical to input_ids **except**: for any chunk that
        contains an `eos_token_id`, all label positions *after* the first
        EOS are set to -100 to prevent loss calculation. This stops the
        model from being trained to predict the start of the next,
        unrelated document.
    6.  `attention_mask` is deliberately omitted.
    """

    def __init__(self, tokenizer, ctx_len: int):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.ctx_len      = ctx_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_ids = []
        for feat in features:
            ids = feat["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)

            if ids.numel() > 0 and ids[0].item() == self.bos_token_id:
                ids = ids[1:]
            
            if ids.numel() > 0 and ids[-1].item() != self.eos_token_id:
                ids = torch.cat([ids, torch.tensor([self.eos_token_id], dtype=torch.long)])
            
            all_ids.append(ids)

        if not all_ids:
            return {
                "input_ids": torch.empty(0, self.ctx_len, dtype=torch.long),
                "labels": torch.empty(0, self.ctx_len, dtype=torch.long),
            }

        all_ids_cat = torch.cat(all_ids, dim=0)

        if all_ids_cat.size(0) < self.ctx_len:
            return {
                "input_ids": torch.empty(0, self.ctx_len, dtype=torch.long),
                "labels": torch.empty(0, self.ctx_len, dtype=torch.long),
            }
            
        chunks = all_ids_cat.unfold(dimension=0, size=self.ctx_len, step=self.ctx_len)
        input_ids = chunks.clone()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,  # (num_chunks, ctx_len)
            "labels": labels,        # (num_chunks, ctx_len)
        }


class DataCollatorShakespeare:
    """
    Collator for small datasets like Shakespeare that uses padding instead of chunking.
    
    This approach is better for small datasets where chunking might lose too much data.
    Each sequence is padded/truncated to ctx_len, and padding tokens are ignored in loss.
    """

    def __init__(self, tokenizer, ctx_len: int):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.ctx_len = ctx_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        
        for feat in features:
            ids = feat["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            
            # Remove BOS if present
            if ids.numel() > 0 and ids[0].item() == self.bos_token_id:
                ids = ids[1:]
            
            # Truncate if too long
            if ids.numel() > self.ctx_len:
                ids = ids[:self.ctx_len]
            
            # Pad if too short
            if ids.numel() < self.ctx_len:
                padding_length = self.ctx_len - ids.numel()
                padding = torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ids = torch.cat([ids, padding])
            
            # Create labels (same as input_ids, but pad tokens become -100)
            labels = ids.clone()
            labels[labels == self.pad_token_id] = -100
            
            batch_input_ids.append(ids)
            batch_labels.append(labels)
        
        return {
            "input_ids": torch.stack(batch_input_ids),  # (batch_size, ctx_len)
            "labels": torch.stack(batch_labels),        # (batch_size, ctx_len)
        }