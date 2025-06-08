import os
from pathlib import Path

import argparse
import importlib.util
import wandb
from torch.utils.data import DataLoader

from transformers import (
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn

from datasets import load_dataset
from loguru import logger
from pprint import pformat

from model import TokenFlowModel, TokenFlowConfig
from dataloader import DataCollatorPretrainFlow


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

    # your paths
    CACHE_DIR  = "/mnt/weka/home/lzchen/bscode/.hf/datasets"
    DISK_TRAIN = "/mnt/weka/home/lzchen/bscode/fineweb_10b/train.arrow"
    DISK_VALID = "/mnt/weka/home/lzchen/bscode/fineweb_10b/valid.arrow"

    # make sure parent exists
    Path(DISK_TRAIN).parent.mkdir(parents=True, exist_ok=True)

    # push HF caches into your scratch
    os.environ["HF_DATASETS_CACHE"]  = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = str(Path(CACHE_DIR).parent / "hf_models")

    # 1) download + shuffle + split (all cached by HF under CACHE_DIR)
    raw = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        cache_dir=CACHE_DIR,
        streaming=False,
    ).shuffle(seed=42)

    split = raw.train_test_split(test_size=0.002, shuffle=False)
    train_raw = split["train"]
    valid_raw = split["test"]

    # 2) tokenize → write to DISK_{TRAIN,VALID}.arrow on first run, skip thereafter
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
        )

    train_data = train_raw.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=16,
        cache_file_name=DISK_TRAIN
    )

    validation_data = valid_raw.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=16,
        cache_file_name=DISK_VALID
    )
    
    data_collator = DataCollatorPretrainFlow(tokenizer=tokenizer, ctx_len=cfg.blk_num*cfg.blk_size, blk_size=cfg.blk_size)

    model_config = TokenFlowConfig(
        blk_num=cfg.blk_num,
        blk_size=cfg.blk_size,
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )

    model = TokenFlowModel(model_config)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        # eval_strategy="steps",
        # eval_steps=cfg.eval_steps,
        bf16=True,
        learning_rate=cfg.learning_rate,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,
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
        torch_compile=True,
        ddp_find_unused_parameters=False,
        save_total_limit=3,
    )
    
    if training_args.process_index == 0:
        train_loader = DataLoader(
            train_data,
            batch_size=4,                  # print 4 examples
            shuffle=True,
            collate_fn=data_collator,
            num_workers=cfg.dataloader_num_workers,
        )

        # grab one batch
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]    # (4, ctx_len)
        labels    = batch["labels"]       # (4, ctx_len)

        # loop and decode
        for i in range(input_ids.size(0)):
            # full context (including padding)
            tokens = input_ids[i].tolist()
            decoded_input = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)

            # only the real tokens (ignore -100 labels)
            label_ids = [tok for tok in labels[i].tolist() if tok != -100]
            decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

            print(f"--- Example {i} ---")
            print("Input IDs  :", tokens)
            print("\n Decoded in :", decoded_input)
            print("\n Decoded lbl:", decoded_labels)
            print()
        model_config.save_pretrained(cfg.output_dir)

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        wandb.init(
            project="TokenFlow",
            name=cfg.run_name,
        )
        logger.info(f"Model Configuration:\n{pformat(model_config.to_dict())}")
        
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
        data_collator=data_collator,
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

