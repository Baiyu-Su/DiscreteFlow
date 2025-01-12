class MyConfig:
    M = 8
    N = 128

    output_dir = "./out"

    # Model config
    vocab_size = 32000
    hidden_size = 3200
    intermediate_size = 3200 * 4
    num_attention_heads = 8
    num_hidden_layers = 6
    max_sequence_length = 2048
    rope_scaling = 10000

    # Model checkpoint for LLaMA embeddings
    llama_checkpoint = "openlm-research/open_llama_3b"

    # T5 checkpoint for tokenizer
    t5_tokenizer_checkpoint = "t5-base"
    
    per_device_train_batch_size = 20
    gradient_accumulation_steps = 18
    
    logging_steps = 50
    dataloader_num_workers = 2
    max_steps = 10000
