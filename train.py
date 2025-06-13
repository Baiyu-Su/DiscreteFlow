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
    TrainerCallback,
)
import torch
import torch.nn as nn

from datasets import load_dataset
from loguru import logger
from pprint import pformat

from model import TokenFlowModel, TokenFlowConfig, RMSNorm, NormalizedEmbedding
from dataloader import DataCollatorPretrainFlow, DataCollatorShakespeare


class NormalizationCallback(TrainerCallback):
    """
    Callback to re-normalize the weights of `NormalizedEmbedding` layers
    at the end of each training step.
    """
    def on_step_end(self, args, state, control, model, **kwargs):
        """
        After the optimizer step, re-normalize the embedding weights.
        """
        for module in model.modules():
            if isinstance(module, NormalizedEmbedding):
                with torch.no_grad():
                    module._normalize_weights()


class StatsLoggingCallback(TrainerCallback):
    """
    Logs, every logging step:
      • RMS & max‑abs of every nn.LayerNorm / RMSNorm weight (layernorms/…)
      • RMS & max‑abs of model_logits & singular_logits (logits/…)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._grad_stats = {}

        # A closure that captures the parameter name and stores its grad stats
        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    # This hook is called during the backward pass
                    detached_grad = grad.detach()
                    self._grad_stats[f"grads/{name}_rms"] = detached_grad.pow(2).mean().sqrt().item()
                    self._grad_stats[f"grads/{name}_max"] = detached_grad.abs().max().item()
            return hook

        # Register the hook for each relevant parameter
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, RMSNorm, NormalizedEmbedding)):
                if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                    # The hook will be named after the module's weight parameter
                    param_name = f"{name}.weight"
                    module.weight.register_hook(make_hook(param_name))
                elif hasattr(module, 'raw_weight') and module.raw_weight is not None and module.raw_weight.requires_grad:
                    # The hook will be named after the module's weight parameter
                    param_name = f"{name}.raw_weight"
                    module.raw_weight.register_hook(make_hook(param_name))

    # on_train_batch_end is no longer needed for gradients
    # def on_train_batch_end(...):

    def on_step_end(self, args, state, control, **kwargs):
        # Respect the Trainer's should_log and only on rank 0
        if not control.should_log or args.process_index != 0:
            return

        logs = {}

        # Log logits stats if they exist
        logits_stats = getattr(self.model, "_logits_stats", None)
        if logits_stats is not None:
            logs.update(logits_stats)

        # Log layernorm weight stats
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, RMSNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.data
                    logs[f"layernorms/{name}_rms"] = w.pow(2).mean().sqrt().item()
                    logs[f"layernorms/{name}_max"] = w.abs().max().item()

        # Log the gradient stats collected by the hooks and then clear them
        if self._grad_stats:
            logs.update(self._grad_stats)
            self._grad_stats = {}

        if logs:
            wandb.log(logs, step=state.global_step)


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
        "--dataset",
        type=str,
        default="fineweb",
        choices=["fineweb", "shakespeare"],
        help="Dataset to use for training (fineweb or shakespeare)"
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Use causal attention (default: False)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory from config file"
    )

    args = parser.parse_args()

    # Dynamically import the config file
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    cfg = config_module.MyConfig
    
    # Override output_dir if provided via command line
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    
    # Redirect all caches to scratch directory
    CACHE_DIR = "/u/chizhang/scratch/data/.hf/datasets"
    HF_CACHE_DIR = "/u/chizhang/scratch/data/.hf"
    DISK_TRAIN = f"/u/chizhang/scratch/data/{args.dataset}/train.arrow"
    DISK_VALID = f"/u/chizhang/scratch/data/{args.dataset}/valid.arrow"

    # make sure parent exists
    Path(DISK_TRAIN).parent.mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # push HF caches into your scratch
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/models"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"
    
    # Use dataset-appropriate tokenizer
    if args.dataset == "shakespeare":
        # Use LLaMA tokenizer for Shakespeare (matches teacher model)
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    elif args.dataset == "fineweb":
        # Use GPT-2 tokenizer for FineWeb (matches GPT-2 teacher model)
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 1) download + shuffle + split (all cached by HF under CACHE_DIR)
    if args.dataset == "fineweb":
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
        
    elif args.dataset == "shakespeare":
        raw = load_dataset(
            "tiny_shakespeare",
            split="train",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        # Join all text and split into chunks for better training
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
        
        train_chunks = chunks[:n_train] if n_train > 0 else chunks[:1]  # At least 1 for training
        valid_chunks = chunks[n_train:] if n_train < n_chunks else chunks[-1:]  # At least 1 for validation
        
        # Convert back to dataset format
        from datasets import Dataset
        train_raw = Dataset.from_dict({"text": train_chunks})
        valid_raw = Dataset.from_dict({"text": valid_chunks})
        
        print(f"Shakespeare dataset processing:")
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

    print(f"Final dataset sizes:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(validation_data)}")

    # Choose appropriate data collator based on dataset
    if args.dataset == "shakespeare":
        data_collator = DataCollatorShakespeare(tokenizer=tokenizer, ctx_len=cfg.ctx_len)
    else:
        data_collator = DataCollatorPretrainFlow(tokenizer=tokenizer, ctx_len=cfg.ctx_len)

    model_config = TokenFlowConfig(
        ctx_len=cfg.ctx_len,
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        tie_word_embeddings=cfg.tie_word_embeddings,
        load_stats=cfg.load_stats,
        use_causal=args.causal,
        use_gumbel_flow=getattr(cfg, 'use_gumbel_flow', False),
        teacher_model_name=getattr(cfg, 'teacher_model_name', None),
    )

    model = TokenFlowModel(model_config)

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
        compute_metrics=None,
        callbacks=[StatsLoggingCallback(model), NormalizationCallback()],
    )

    if training_args.process_index == 0:
        logger.info("Let the training begin.")
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    if training_args.process_index == 0:
        wandb.finish()


if __name__ == '__main__':
    main()
