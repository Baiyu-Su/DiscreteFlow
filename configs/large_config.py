class MyConfig:
    blk_num = 16
    blk_size = 128

    output_dir = "./out_large_128"
    run_name = "large_run_128"
    load_stats = True

    # Model config
    vocab_size = 128256
    dim = 2048
    n_heads = 16 # Head dim must be divisible by 128
    n_layers = 16
    tie_word_embeddings = True
    
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    
    dataloader_num_workers = 32
    max_steps = 40000
    eval_steps = 1000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 2e-4
    adam_beta2 = 0.95
    weight_decay = 0.01   
         
    warmup_steps = 2000
    lr_scheduler_type = "cosine"
    
    