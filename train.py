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

from model import TokenFlowModel, TokenFlowConfig
from dataloader import DataCollatorFlow


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
    cfg = config_module.MyConfig

    
    tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")

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
    
    train_data = tokenized_dataset["train"]
    validation_data = tokenized_dataset["validation"]
    
    data_collator_flow = DataCollatorFlow(
        tokenizer=tokenizer,
        ctx_len=cfg.M*cfg.N,
        device="cuda"
    )

    model_config = TokenFlowConfig(
        is_inference=False,
        M=cfg.M,
        N=cfg.N,
        vocab_size=32000,
        dim=cfg.dim,
        time_dim=cfg.time_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )

    model = TokenFlowModel(model_config)

    wandb.init(
        project="TokenFlow",
        name=cfg.run_name,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
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
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator_flow,
    )
    
    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()


