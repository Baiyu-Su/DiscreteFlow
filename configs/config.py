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

    # Model checkpoint for LLaMA embeddings
    llama_checkpoint = "openlm-research/open_llama_3b"   # Example

    # T5 checkpoint for tokenizer
    t5_tokenizer_checkpoint = "t5-base"
    
    per_device_train_batch_size = 20
    gradient_accumulation_steps=18
    
    logging_steps=50
    dataloader_num_workers=2
