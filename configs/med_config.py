class MyConfig:
    M = 8
    N = 128

    output_dir = "./out"
    run_name = "test_run"

    # Model config
    vocab_size = 32000
    hidden_size = 1024
    intermediate_size = 5632
    num_attention_heads = 16
    num_hidden_layers = 12
    max_sequence_length = 2048
    rope_scaling = 10000

    # Model checkpoint for LLaMA embeddings
    llama_checkpoint = "keeeeenw/MicroLlama"   # Example

    per_device_train_batch_size = 30
    gradient_accumulation_steps = 4
    
    dataloader_num_workers = 16
    warmup_steps = 1000
    max_steps = 20000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.95
    lr_scheduler_type = "cosine"
