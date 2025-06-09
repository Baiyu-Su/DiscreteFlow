class MyConfig:
    blk_num = 8
    blk_size = 128

    output_dir = "./out_small_norescale"
    run_name = "small_run"
    load_stats = True

    # Model config
    vocab_size = 32000
    dim = 768
    n_heads = 6 # Head dim must be divisible by 128
    n_layers = 12
    tie_word_embeddings = True
    
    per_device_train_batch_size = 40
    gradient_accumulation_steps = 4
    
    dataloader_num_workers = 8
    max_steps = 16000
    eval_steps = 500
    save_steps = 1000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.98
    weight_decay = 0.02   
         
    warmup_steps = 1600
    lr_scheduler_type = "cosine"
    
    