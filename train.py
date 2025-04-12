import os
import json
import multiprocessing

import argparse
import importlib.util
import wandb

from transformers import (
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
)
import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import load_from_disk
from loguru import logger
from pprint import pformat
from dataclasses import asdict

from model import TokenFlowModel, TokenFlowConfig
from dataloader import DataCollatorFlow


def load_pretrained_embedding(cfg, model):
    base_model = AutoModel.from_pretrained(
        cfg.llama_checkpoint,
        trust_remote_code=True,
        device_map="cpu"  # load on CPU to avoid GPU memory spike
    )
    pretrained_token_embedding = base_model.get_input_embeddings()
    with torch.no_grad():
        model.token_embed.weight[:32000, :] = pretrained_token_embedding.weight[:32000, :]
        nn.init.normal_(model.token_embed.weight[32000, :], mean=0.0, std=0.02)

    del base_model
    del pretrained_token_embedding
    return model


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
    cfg = config_module.MyConfig
    
    tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        cache_dir="./.hf_cache",
        trust_remote_code=True,
    )
    validation_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
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

    train_tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=16,
    )
    validation_tokenized_dataset = validation_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=16,
    )
    
    train_data = train_tokenized_dataset
    validation_data = validation_tokenized_dataset
    
    data_collator_flow = DataCollatorFlow(tokenizer=tokenizer, ctx_len=cfg.M*cfg.N, block_size=cfg.N)

    model_config = TokenFlowConfig(
        is_inference=False,
        M=cfg.M,
        N=cfg.N,
        vocab_size=32001,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )

    model = TokenFlowModel(model_config)
    model = load_pretrained_embedding(cfg, model)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        bf16=True,
        learning_rate=cfg.learning_rate,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        eval_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=False,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        save_total_limit=3,
    )
    
    if training_args.process_index == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        config_path = os.path.join(cfg.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(model_config), f, indent=2)

        wandb.init(
            project="TokenFlow",
            name=cfg.run_name,
        )
        logger.info(f"Model Configuration:\n{pformat(asdict(model_config))}")
        
        total_params = sum(p.numel() for p in model.parameters())
        embed_params = sum(p.numel() for p in model.token_embed.parameters())
        nonembed_params = total_params - embed_params

        logger.info(f"HuggingFace Training Arguments:\n{training_args}")
        logger.info(f"Total model parameters: {total_params}")
        logger.info(f"Non-embedding parameters: {nonembed_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator_flow,
    )

    if training_args.process_index == 0:
        logger.info("The training begins.")
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    if training_args.process_index == 0:
        wandb.finish()


if __name__ == '__main__':
    main()


