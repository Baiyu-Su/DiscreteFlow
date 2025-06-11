import torch
from typing import List, Dict, Any


class DataCollatorPretrainFlow:
    """
    Collator for block-wise pre-training with right-hand EOS padding.

    Rules implemented
    ------------------
    1.  Truncate sequences longer than ctx_len; right-pad shorter ones
        with `eos_token_id` until they reach exactly ctx_len.
    2.  Labels are identical to input_ids (no AR shift) **except**:
        after the block that contains the first padded EOS, all
        positions are set to -100 so they contribute no loss.
    3.  `attention_mask` is deliberately omitted.
    """

    def __init__(self, tokenizer, ctx_len: int, blk_size: int):
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.ctx_len      = ctx_len
        self.blk_size     = blk_size

    def _right_pad(self, ids: torch.Tensor) -> torch.Tensor:
        """Truncate / right-pad to ctx_len using eos."""
        if ids.size(0) > self.ctx_len:                      # truncate
            ids = ids[: self.ctx_len]
        elif ids.size(0) < self.ctx_len:                    # pad
            pad_len = self.ctx_len - ids.size(0)
            pad     = ids.new_full((pad_len,), self.eos_token_id)
            ids     = torch.cat([ids, pad], dim=0)
        return ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_labels = [], []

        for feat in features:
            # 1) get tensor
            ids = feat["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
                
            if ids.numel() > 0 and ids[0].item() == self.bos_token_id:
                ids = ids[1:]

            ids = self._right_pad(ids)

            labels = ids.clone()

            eos_mask = labels == self.eos_token_id
            if eos_mask.any():
                first_eos = eos_mask.nonzero(as_tuple=True)[0][0].item()
                labels[first_eos + 1 :] = -100

            batch_input_ids.append(ids)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),   # (B, ctx_len)
            "labels": torch.stack(batch_labels, dim=0),  # (B, ctx_len)
        }
