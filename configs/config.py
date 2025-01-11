# configs/my_config.py

class MyConfig:
    M = 8
    N = 128

    # Model config
    vocab_size = 32000
    hidden_size = 1024
    intermediate_size = 4096
    num_attention_heads = 16
    num_hidden_layers = 12
    max_sequence_length = 4096
    rope_scaling = 10000

    # Data paths
    train_data = "c4"
    # Or any other relevant path

    # Model checkpoint for LLaMA embeddings
    llama_checkpoint = "openlm-research/open_llama_3b"   # Example
    # Or meta-llama/Llama-2-7b-hf, etc.

    # T5 checkpoint for tokenizer
    t5_tokenizer_checkpoint = "t5-base"
