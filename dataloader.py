import torch

class DataCollatorFlow:
    def __init__(self, tokenizer, ctx_len):
        """
        :param tokenizer: Tokenizer to handle padding.
        :param ctx_len: Desired fixed context length.
        """
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len

    def __call__(self, features):
        """
        Args:
          features: list of dicts from the tokenized dataset, each with 'input_ids'
        
        Returns:
          Dict with:
            - 'input_ids': Tensor of shape (batch_size, ctx_len)
            - 'labels': Tensor of shape (batch_size, ctx_len) where tokens that were padded are set to -100.
        """
        batch_input_ids = []
        batch_labels = []
        
        for f in features:
            # Retrieve the original token ids for this example.
            ids = f["input_ids"]
            orig_len = len(ids)

            # Double guarantee no sequence is longer than ctx_len.
            if orig_len > self.ctx_len:
                ids = ids[:self.ctx_len]
                orig_len = self.ctx_len

            pad_len = self.ctx_len - orig_len

            # Pad the input_ids with pad_token_id.
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_len
            batch_input_ids.append(padded_ids)

            # Ignore all padding tokens.
            labels = ids + [-100] * pad_len
            batch_labels.append(labels)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }
