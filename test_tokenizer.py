from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")

print("Special tokens map:", tokenizer.special_tokens_map)
print("All special tokens:", tokenizer.all_special_tokens)
print("All special token ids:", tokenizer.all_special_ids)
# print("Pad token id:", tokenizer.pad_id)

print("Vocab size:", tokenizer.vocab_size)

