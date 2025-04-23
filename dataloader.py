import torch
import torch.nn.functional as F

class DataCollatorPretrainFlow:
    def __init__(self, tokenizer, ctx_len, N):
        """
        :param tokenizer: Tokenizer to handle padding.
        :param ctx_len: Desired fixed context length.
        :param N: Block size.
        """
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.ctx_len = ctx_len
        self.N = N
        self.max_seq_len = ctx_len - N

    def __call__(self, features):
        """
        Args:
          features: list of dicts from the tokenized dataset, each with 'input_ids'
        Returns:
          Dict with:
            - 'input_ids': Tensor of shape (batch_size, ctx_len)
            - 'labels':    Tensor of shape (batch_size, ctx_len), where any pad (post-) is -100.
            - 'attention_mask': Tensor of shape (batch_size, ctx_len)
        """
        batch_input_ids = []
        batch_labels = []

        for f in features:
            # 1) pull out the ids tensor
            ids = f["input_ids"] if isinstance(f, dict) else f
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)

            # 2) truncate if too long
            if ids.size(0) > self.max_seq_len:
                ids = ids[: self.max_seq_len]

            # 3) figure out how many eos tokens to add so len % N == 0 (and at least one)
            rem = ids.size(0) % self.N
            num_eos = self.N - rem
            eos_ids = torch.full(
                (num_eos,),
                fill_value=self.eos_token_id,
                dtype=torch.long
            )
            ids = torch.cat([ids, eos_ids], dim=0)

            # 4) pad on the right up to ctx_len
            pad_len = self.ctx_len - ids.size(0)
            if pad_len > 0:
                ids = F.pad(
                    ids,
                    pad=(0, pad_len),
                    value=self.pad_token_id
                )

            # 5) make labels, masking pads
            labels = ids.clone()
            labels[labels == self.pad_token_id] = -100

            batch_input_ids.append(ids)
            batch_labels.append(labels)

        # stack into batch
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_labels    = torch.stack(batch_labels,    dim=0)

        # 6) attention mask: 1 for real tokens (including eos), 0 for pad
        attention_mask = (batch_input_ids != self.pad_token_id).long()

        return {
            "input_ids":      batch_input_ids,
            "labels":         batch_labels,
            "attention_mask": attention_mask,
        }