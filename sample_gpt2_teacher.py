"""sample_gpt2_teacher.py — Sample text from fine-tuned GPT-2 Shakespeare model

Usage
-----
$ python sample_gpt2_teacher.py --model_path /u/chizhang/scratch/data/gpt2_shakespeare_teacher

Loads the fine-tuned GPT-2 model and generates Shakespeare-style text samples.
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

def parse_args():
    """CLI helper."""
    parser = argparse.ArgumentParser(description="Sample from fine-tuned GPT-2 Shakespeare model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/u/chizhang/scratch/data/gpt2_shakespeare_teacher",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of text samples to generate")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="", 
        help="Starting prompt (leave empty for unconditional generation)"
    )
    parser.add_argument(
        "--eval_perplexity", 
        action="store_true", 
        help="Evaluate perplexity on Shakespeare validation set"
    )
    parser.add_argument("--max_eval_length", type=int, default=256, help="Maximum sequence length for perplexity evaluation")
    return parser.parse_args()


def prepare_shakespeare_eval_data(tokenizer, max_length=256):
    """Prepare Shakespeare validation dataset for perplexity evaluation."""
    
    # Set up cache directories (same as training code)
    CACHE_DIR = "/u/chizhang/scratch/data/.hf/datasets"
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    
    # Load tiny Shakespeare dataset
    raw = load_dataset(
        "tiny_shakespeare",
        split="train",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    
    # Same sliding window approach as training code
    full_text = "\n".join(raw["text"])
    chunk_size = max_length
    overlap = max_length - 1  # Maximum overlap (stride = 1)
    
    chunks = []
    for i in range(0, len(full_text) - chunk_size + 1, chunk_size - overlap):
        chunk = full_text[i:i + chunk_size]
        chunks.append(chunk)
    
    if len(full_text) > chunk_size:
        chunks.append(full_text[-(chunk_size):])
    
    # Use same split as training (90% train, 10% val)
    split_idx = int(len(chunks) * 0.9)
    val_chunks = chunks[split_idx:]  # Only use validation chunks
    
    print(f"Loaded {len(val_chunks)} validation chunks for perplexity evaluation")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
    
    # Create validation dataset
    val_dataset = Dataset.from_dict({"text": val_chunks})
    
    # Tokenize
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    
    return val_tokenized


def calculate_perplexity(model, tokenizer, eval_dataset, device, batch_size=8):
    """Calculate perplexity on the evaluation dataset."""
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print("Calculating perplexity...")
    
    # Process in batches
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        batch_data = eval_dataset[i:i+batch_size]
        
        # Prepare batch
        input_ids_list = batch_data["input_ids"]
        
        # Convert to tensors and pad
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_batch = []
        attention_mask_batch = []
        
        for ids in input_ids_list:
            # Pad sequence
            padded_ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            attention_mask = [1] * len(ids) + [0] * (max_len - len(ids))
            
            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(attention_mask)
        
        input_ids = torch.tensor(input_ids_batch).to(device)
        attention_mask = torch.tensor(attention_mask_batch).to(device)
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        # Set labels to -100 for padded tokens (they won't contribute to loss)
        labels[attention_mask == 0] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Count only non-padded tokens
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    args = parse_args()
    
    # Set HuggingFace cache to scratch directory
    HF_CACHE_DIR = "/u/chizhang/scratch/models/.hf"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR + "/transformers"
    
    print(f"Loading model from: {args.model_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Max length: {args.max_length}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 50)
    
    # Load the fine-tuned model and tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Set to evaluation mode
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print("=" * 50)
    
    # Calculate perplexity if requested
    if args.eval_perplexity:
        print("\n🔍 Evaluating perplexity on Shakespeare validation set...")
        print("-" * 50)
        
        # Load evaluation dataset
        eval_dataset = prepare_shakespeare_eval_data(tokenizer, max_length=args.max_eval_length)
        
        # Calculate perplexity
        avg_loss, perplexity = calculate_perplexity(model, tokenizer, eval_dataset, device)
        
        print(f"\n📊 Perplexity Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print("=" * 50)
    
    # Generate samples
    for i in range(args.num_samples):
        print(f"\n📝 Sample {i+1}:")
        print("-" * 30)
        
        # Prepare input
        if args.prompt:
            input_text = args.prompt
            print(f"Prompt: '{input_text}'")
            print(f"Generated continuation:")
        else:
            input_text = ""
            print("Unconditional generation:")
        
        # Tokenize input
        if input_text:
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        else:
            # Start with BOS token for unconditional generation
            input_ids = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id]]).to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # If we had a prompt, show only the generated part
        if args.prompt:
            generated_part = generated_text[len(input_text):]
            print(f"{generated_part}")
        else:
            print(f"{generated_text}")
        
        print("-" * 30)
    
    print("\n🎭 Generation complete!")
    
    # Some suggested prompts for Shakespeare
    print("\n💡 Try some of these prompts:")
    suggested_prompts = [
        "To be or not to be,",
        "Romeo, Romeo, wherefore art thou",
        "All the world's a stage,",
        "Once upon a midnight dreary,",
        "When shall we three meet again?",
        "Fair is foul, and foul is fair:",
        "But soft, what light through yonder window breaks?"
    ]
    
    for prompt in suggested_prompts:
        print(f"  python sample_gpt2_teacher.py --model_path {args.model_path} --prompt \"{prompt}\"")


if __name__ == "__main__":
    main() 