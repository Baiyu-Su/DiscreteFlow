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