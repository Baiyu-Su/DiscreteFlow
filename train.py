import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import argparse
import importlib.util

from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    AutoModel,         # We will extract the embedding layer from here
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import datasets
from datasets import load_dataset
from loguru import logger

from model import DiscreteFlowModel, DiscreteFlowConfig


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t in [0, 1] to an embedding of dimension embed_dim.
    """
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # A small MLP after the sinusoidal. Do I really need this??
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor):
        """
        t: shape (batch_size, M) or (batch_size, 1) with values in [0,1].
        Returns shape: (batch_size, M, embed_dim)
        """
        # We'll create sinusoidal embeddings of dimension embed_dim
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            torch.linspace(
                0,
                math.log(self.max_period),
                steps=half_dim,
                device=t.device
            )
        )
        # shape of freqs = (half_dim,) – broadcast against t
        # Expand t to match for broadcast
        t = t.unsqueeze(-1)  # shape: (batch_size, M, 1)
        # Angular velocity
        args = t * freqs  # shape: (batch_size, M, half_dim)

        sin_cos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            # If embed_dim is odd, we can pad by one
            sin_cos = F.pad(sin_cos, (0,1,0,0,0,0))  # pad last dim by 1

        return self.proj(sin_cos)  # (batch_size, M, embed_dim)

class DataCollatorFlow:
    def __init__(
        self,
        pretrained_token_embedding,
        time_embedding_module,
        tokenizer,
        M=8,
        N=128
    ):
        """
        :param pretrained_token_embedding: The frozen input embedding module.
        :param time_embedding_module: A module that maps t in [0,1] to embed_dim.
        :param tokenizer: For potential padding if needed, though we do custom chunking.
        :param M: number of blocks
        :param N: block size (N-1 text tokens + 1 time token)
        """
        self.token_embedding = pretrained_token_embedding
        self.time_embedding_module = time_embedding_module
        self.tokenizer = tokenizer
        self.M = M
        self.N = N
        self.block_text_len = N - 1  # 127

        # Precompute for convenience
        self.context_length = self.M * self.block_text_len  # 8 * 127 = 1016
        self.embed_dim = self.token_embedding.weight.shape[1]  # dimension of word embedding

    def __call__(self, features):
        """
        features: a list of dicts from the tokenized dataset,
                  each dict has 'input_ids', etc.

        We return a dict with:
          - 'input_embeddings': shape (batch_size, 2*(M*N), embed_dim)
          - 'label_ids'
        """

        batch_input_ids = []
        for f in features:
            # We just take input_ids
            print(f)
            ids = f["input_ids"]
            # Truncate or pad to exactly self.context_length
            if len(ids) >= self.context_length:
                ids = ids[:self.context_length]
            else:
                # pad with e.g. tokenizer.pad_token_id
                pad_len = self.context_length - len(ids)
                ids = ids + [self.tokenizer.pad_token_id] * pad_len
            batch_input_ids.append(ids)

        # Convert to tensor: (batch_size, context_length)
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)

        # Step 1: get X1 = clean token embeddings
        # shape => (batch_size, context_length, embed_dim)
        with torch.no_grad():
            X1 = self.token_embedding(input_ids_tensor)

        B = X1.size(0)
        # Reshape to (batch_size, M, (N-1), embed_dim) for block-wise processing
        X1 = X1.view(B, self.M, self.block_text_len, self.embed_dim)

        # Step 2: sample t ~ Uniform(0, 1) for each block => shape (B, M)
        t_values = torch.rand((B, self.M))

        # Step 3: sample noise X0 => same shape as X1 => (B, context_length, embed_dim)
        X0 = torch.randn_like(X1)

        # For easier block-wise merges, reshape X0 as well
        X0 = X0.view(B, self.M, self.block_text_len, self.embed_dim)

        # Step 4: perturb to make noisy blocks t: X_t = t * X1 + (1 - t) * X0
        # We have t shaped (B, M). We need to broadcast across (N-1, embed_dim)
        t_values_4d = t_values.view(B, self.M, 1, 1)  # shape => (B, M, 1, 1)
        Xt = t_values_4d * X1 + (1 - t_values_4d) * X0
        # shape => (B, M, N-1, embed_dim)

        # Step 5: append time embedding to each block
        # We embed t_values to shape (B, M, embed_dim)
        # We allow gradients thru time embedding here. If not using NN for time embedding, set no grad
        t_embeddings = self.time_embedding_module(t_values)  # shape (B, M, embed_dim)

        # Xt_blocks: (B, M, (N-1), embed_dim)
        # t_embeddings: (B, M, embed_dim)

        # 1) For PART 2, just cat along dim=2
        part2_blocks = torch.cat(
            [Xt, t_embeddings.unsqueeze(2)],  # => (B, M, 1, embed_dim)
            dim=2
        )  # => shape (B, M, N, embed_dim)

        # Flatten the (M, N) dimensions
        part2 = part2_blocks.view(B, M*N, self.embed_dim)  # (B, M*N, embed_dim)

        # 2) For PART 1, we do the same logic but with t=1 for all blocks
        #    Suppose X1_blocks_reshaped is (B, M, (N-1), embed_dim)
        #    and t1_embeddings is (B, M, embed_dim)
        ones = torch.ones((B, self.M), device=X1.device)
        t1_embeddings = self.time_embedding_module(ones)  # (B, M, embed_dim)
        part1_blocks = torch.cat(
            [X1, t1_embeddings.unsqueeze(2)],  # => (B, M, 1, embed_dim)
            dim=2
        )  # => (B, M, N, embed_dim)

        part1 = part1_blocks.view(B, M*N, self.embed_dim)  # (B, M*N, embed_dim)

        # 3) Finally, concat Part1 and Part2 along dimension=1
        final_input = torch.cat([part1, part2], dim=1)  # => (B, 2*M*N, embed_dim)

        return {
            "input_embeddings": final_input,  # shape [B, 2*(M*N), embed_dim]
            # Return the original tokens as labels
            "labels": input_ids_tensor,       # shape [B, context_length]
        }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/my_config.py",
        help="Path to the Python config file (e.g. configs/my_config.py)"
    )
    args = parser.parse_args()

    # Dynamically import the config file
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    # Now config_module is loaded. We assume it has MyConfig in it.
    cfg = config_module.MyConfig

    
    # 1) Build the T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    # 2) Load the LLaMA (or open-llama) base model to extract embeddings
    base_model = AutoModel.from_pretrained(
        cfg.llama_checkpoint,
        trust_remote_code=True
    )
    # This base_model should have an embedding dimension we can find:
    embed_dim = base_model.config.hidden_size  # e.g. 1024 or 4096, etc.
    print("Embedding dimension:", embed_dim)
    pretrained_token_embedding = base_model.get_input_embeddings()
    # Freeze embedding parameters
    for param in pretrained_token_embedding.parameters():
        param.requires_grad = False


    dataset = load_dataset(
        "allenai/c4", 
        "en", 
        split="train",
        streaming=True,
        cache_dir="./.hf_cache"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt",
            truncation=True,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"])

    print("Sample tokenized data:", next(iter(tokenized_dataset)))

    train_data = tokenized_dataset

    time_embedding_module = SinusoidalTimeEmbedding(embed_dim=embed_dim)

    data_collator_flow = DataCollatorFlow(
        pretrained_token_embedding=pretrained_token_embedding,
        time_embedding_module=time_embedding_module,
        tokenizer=tokenizer,
        M=cfg.M,
        N=cfg.N
    )

    model_config = DiscreteFlowConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_hidden_layers=cfg.num_hidden_layers,
            max_sequence_length=cfg.max_sequence_length,
            rope_scaling=cfg.rope_scaling
        )

    # 6) Build FlowLlamaModel
    model = DiscreteFlowModel(model_config, M=cfg.M, N=cfg.N)


    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="no",
        num_train_epochs=1,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator_flow,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()


