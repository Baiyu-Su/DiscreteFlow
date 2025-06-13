class MyConfig:
    ctx_len = 1024

    output_dir = "./out_small_causal_encoder"
    run_name = "small_run_causal_encoder"
    load_stats = True

    # Model config
    vocab_size = 32000
    embed_dim = 64
    dim = 768
    n_heads = 12 # Head dim must be divisible by 128
    n_layers = 12
    tie_word_embeddings = True
    
    per_device_train_batch_size = 40
    gradient_accumulation_steps = 16
    
    dataloader_num_workers = 8
    max_steps = 20000
    eval_steps = 2000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.98
    weight_decay = 0.02   
         
    warmup_steps = 2000
    lr_scheduler_type = "cosine"
    
    