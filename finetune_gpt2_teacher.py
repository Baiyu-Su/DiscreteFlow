"""finetune_gpt2_teacher.py — Fine-tune GPT-2 on Shakespeare as teacher model

Usage
-----
$ python finetune_gpt2_teacher.py --model gpt2 --output_dir checkpoints/gpt2_shakespeare_teacher

Fine-tunes GPT-2 on Shakespeare to create a better teacher model for DiscreteFlow.
"""

import os
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from tqdm import tqdm

def parse_args():
    """CLI helper."""
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Shakespeare")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model size to fine-tune"
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/gpt2_shakespeare_teacher", help="Output directory")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    return parser.parse_args()


def prepare_shakespeare_data(tokenizer, max_length=256, train_split=0.9):
    """Prepare Shakespeare dataset for fine-tuning."""
    
    # Set up cache directories (same as existing code)
    CACHE_DIR = "/u/chizhang/scratch/data/.hf/datasets"
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    
    # Load tiny Shakespeare dataset
    raw = load_dataset(
        "tiny_shakespeare",
        split="train",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    
    # Same sliding window approach as existing training code
    full_text = "\n".join(raw["text"])
    chunk_size = max_length
    overlap = max_length - 1  # Maximum overlap (stride = 1)
    
    chunks = []
    for i in range(0, len(full_text) - chunk_size + 1, chunk_size - overlap):
        chunk = full_text[i:i + chunk_size]
        chunks.append(chunk)
    
    if len(full_text) > chunk_size:
        chunks.append(full_text[-(chunk_size):])
    
    print(f"Shakespeare dataset processing:")
    print(f"  Full text length: {len(full_text):,} characters")
    print(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"  Total chunks created: {len(chunks)}")
    
    # Split into train/val
    split_idx = int(len(chunks) * train_split)
    train_chunks = chunks[:split_idx]
    val_chunks = chunks[split_idx:]
    
    print(f"  Train chunks: {len(train_chunks)}")
    print(f"  Val chunks: {len(val_chunks)}")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_chunks})
    val_dataset = Dataset.from_dict({"text": val_chunks})
    
    # Tokenize
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    
    return train_tokenized, val_tokenized


def main():
    args = parse_args()
    
    # Set HuggingFace cache to scratch directory (same as eval script)
    HF_CACHE_DIR = "/u/chizhang/scratch/models/.hf"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR + "/transformers"
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(HF_CACHE_DIR + "/transformers", exist_ok=True)
    
    print(f"Fine-tuning {args.model} on Shakespeare dataset")
    print(f"Output directory: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"HF Cache: {HF_CACHE_DIR}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading {args.model}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Prepare datasets
    print("Preparing Shakespeare datasets...")
    train_dataset, val_dataset = prepare_shakespeare_data(
        tokenizer, 
        max_length=args.max_length
    )
    print()
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard
        dataloader_pin_memory=False,  # Avoid memory issues
        gradient_checkpointing=True,  # Save memory
        fp16=True,  # Use mixed precision
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate final perplexity
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    
    print("=" * 50)
    print(f"Final Results:")
    print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print("=" * 50)
    
    print(f"\nFine-tuned model saved to: {args.output_dir}")
    print("You can now use this as a teacher model in your DiscreteFlow training!")


if __name__ == "__main__":
    main() 