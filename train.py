import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import argparse
import importlib.util
import wandb

from torch.utils.data import DataLoader

from transformers import (
    LlamaTokenizer,
    AutoModel,         # We will extract the embedding layer from here
    TrainingArguments,
    Trainer
)
import datasets
from datasets import load_dataset
from loguru import logger

from model import DiscreteFlowModel, DiscreteFlowConfig
from dataloader import SinusoidalTimeEmbedding, DataCollatorFlow


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/my_config.py",
        help="Path to the Python config file (e.g. configs/my_config.py)"
    )
    parser.add_argument(
        "--preprocessed_data_dir",
        type=str,
        default="./preprocessed_c4",
        help="Path to the offline tokenized dataset"
    )
    args = parser.parse_args()

    # Dynamically import the config file
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    # Now config_module is loaded. We assume it has MyConfig in it.
    cfg = config_module.MyConfig

    
    tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the LLaMA base model to extract embeddings
    base_model = AutoModel.from_pretrained(
        cfg.llama_checkpoint,
        trust_remote_code=True
    )

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
        cache_dir="./.hf_cache",
        trust_remote_code=True,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=1024, 
            return_tensors=None,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=16,
    )
    
    train_data = tokenized_dataset

    time_embedding_module = SinusoidalTimeEmbedding(embed_dim=embed_dim)

    data_collator_flow = DataCollatorFlow(
        pretrained_token_embedding=pretrained_token_embedding,
        time_embedding_module=time_embedding_module,
        tokenizer=tokenizer,
        M=cfg.M,
        N=cfg.N,
        device="cuda"
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

    wandb.init(
        project="DiscreteFlow",
        name=cfg.run_name,
    )


    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        bf16=True,
        learning_rate=cfg.learning_rate,
        adam_beta2=cfg.adam_beta2,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        remove_unused_columns=False,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator_flow,
    )
    
    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()


