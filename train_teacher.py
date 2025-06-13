import os
from pathlib import Path

import argparse
import importlib.util
import wandb
from torch.utils.data import DataLoader

from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch

from datasets import load_dataset
from loguru import logger
from pprint import pformat

from dataloader import DataCollatorPretrainFlow, DataCollatorShakespeare


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/teacher_config.py",
        help="Path to the Python config file (e.g. configs/teacher_config.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        choices=["shakespeare"],
        help="Dataset to use for training (currently only shakespeare supported)"
    )
    args = parser.parse_args()

    # Dynamically import the config file
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    cfg = config_module.MyConfig
    
    # Redirect all caches to scratch directory
    CACHE_DIR = "/u/chizhang/scratch/data/.hf/datasets"
    HF_CACHE_DIR = "/u/chizhang/scratch/data/.hf"
    DISK_TRAIN = f"/u/chizhang/scratch/data/{args.dataset}_teacher/train.arrow"
    DISK_VALID = f"/u/chizhang/scratch/data/{args.dataset}_teacher/valid.arrow"

    # make sure parent exists
    Path(DISK_TRAIN).parent.mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # push HF caches into your scratch
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/models"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"
    
    # Use HuggingFace LLaMA tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    
    # Set pad token to eos token for LLaMA
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) Load and process Shakespeare dataset
    if args.dataset == "shakespeare":
        raw = load_dataset(
            "tiny_shakespeare",
            split="train",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        # Same sliding window approach as TokenFlow
        full_text = "\n".join(raw["text"])
        chunk_size = cfg.ctx_len      # Use context length as chunk size
        overlap = cfg.ctx_len - 1     # Maximum overlap (stride = 1)
        
        chunks = []
        for i in range(0, len(full_text) - chunk_size + 1, chunk_size - overlap):
            chunk = full_text[i:i + chunk_size]
            chunks.append(chunk)
        
        if len(full_text) > chunk_size:
            chunks.append(full_text[-(chunk_size):])
        
        # Split chunks into train/validation (90/10)
        n_chunks = len(chunks)
        n_train = int(n_chunks * 0.9)
        
        train_chunks = chunks[:n_train] if n_train > 0 else chunks[:1]
        valid_chunks = chunks[n_train:] if n_train < n_chunks else chunks[-1:]
        
        # Convert back to dataset format
        from datasets import Dataset
        train_raw = Dataset.from_dict({"text": train_chunks})
        valid_raw = Dataset.from_dict({"text": valid_chunks})
        
        print(f"Teacher model - Shakespeare dataset processing:")
        print(f"  Full text length: {len(full_text):,} characters")
        print(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
        print(f"  Total chunks created: {len(chunks)}")
        print(f"  Training chunks: {len(train_chunks)}")
        print(f"  Validation chunks: {len(valid_chunks)}")
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 2) tokenize → write to DISK_{TRAIN,VALID}.arrow on first run, skip thereafter
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.ctx_len,
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

    print(f"Teacher model - Final dataset sizes:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(validation_data)}")

    # Use standard HuggingFace data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Create LLaMA model configuration
    model_config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=cfg.hidden_dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        max_position_embeddings=cfg.ctx_len,
        rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=cfg.tie_word_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create LLaMA model
    model = LlamaForCausalLM(model_config)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
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
        torch_compile=False,
        ddp_find_unused_parameters=False,
        save_total_limit=3,
    )
    
    if training_args.process_index == 0:
        # Print some examples
        train_loader = DataLoader(
            train_data,
            batch_size=4,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=cfg.dataloader_num_workers,
        )

        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for i in range(input_ids.size(0)):
            tokens = input_ids[i].tolist()
            decoded_input = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            label_ids = [tok for tok in labels[i].tolist() if tok != -100]
            decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

            print(f"--- Teacher Example {i} ---")
            print("Input IDs  :", tokens[:50], "..." if len(tokens) > 50 else "")
            print("\n Decoded in :", decoded_input[:200], "..." if len(decoded_input) > 200 else "")
            print("\n Decoded lbl:", decoded_labels[:200], "..." if len(decoded_labels) > 200 else "")
            print()

        model_config.save_pretrained(cfg.output_dir)

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        wandb.init(
            project="TokenFlow-Teacher",
            name=cfg.run_name,
        )
        
        logger.info(f"Teacher Model Configuration:\n{pformat(model_config.to_dict())}")
        
        total_params = sum(p.numel() for p in model.parameters())
        embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
        nonembed_params = total_params - embed_params

        logger.info(f"HuggingFace Training Arguments:\n{training_args}")
        logger.info(f"Total teacher model parameters: {total_params}")
        logger.info(f"Non-embedding parameters: {nonembed_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator,
        compute_metrics=None,
    )

    if training_args.process_index == 0:
        logger.info("Let the teacher training begin.")
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    
    if training_args.process_index == 0:
        wandb.finish()


if __name__ == '__main__':
    main() 