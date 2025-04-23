import torch
import random
import math

class DataCollatorFlow:
    def __init__(self, tokenizer, ctx_len, block_size):
        """
        :param tokenizer: HuggingFace tokenizer.
        :param ctx_len: Fixed context length (maximum sequence length).
        :param block_size: N for the 'double-sided' random pre-padding block size.
        """
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.block_size = block_size

        if self.tokenizer.bos_token_id is None:
            raise ValueError("tokenizer has no bos_token_id. Please adapt or define one.")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id. Please adapt or define one.")

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def __call__(self, features):
        """
        Args:
            features (List[Dict]): Each element is typically a dict from the HF dataset
                                   containing 'input_ids' (and possibly other fields).

        Returns:
            Dict[str, torch.Tensor]: A batch with 'input_ids' and 'labels' of shape
                                     (batch_size, ctx_len).
        """
        batch_input_ids = []
        batch_labels = []

        for f in features:
            ids = f["input_ids"]

            new_ids = [self.bos_id] + ids + [self.eos_id]
            seq_len = len(new_ids)

            if seq_len > self.ctx_len:
                new_ids = new_ids[:self.ctx_len]
                seq_len = len(new_ids)

            max_r = min(self.block_size - 1, self.ctx_len - seq_len)
            r = random.randint(0, max_r)
            pre_padded_ids = [self.pad_id] * r + new_ids
            seq_len2 = len(pre_padded_ids)

            post_pad_len = self.ctx_len - seq_len2
            final_ids = pre_padded_ids + [self.pad_id] * post_pad_len

            m = math.ceil(seq_len2 / self.block_size)
            labeled_length = m * self.block_size

            labels = []
            for i in range(self.ctx_len):
                if i < labeled_length:
                    labels.append(final_ids[i])
                else:
                    labels.append(-100)

            batch_input_ids.append(final_ids)
            batch_labels.append(labels)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }
