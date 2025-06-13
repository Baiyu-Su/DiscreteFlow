class MyConfig:
    ctx_len = 512

    output_dir = f"/u/chizhang/scratch/data/out_fineweb_tokenflow_ctx{ctx_len}"
    run_name = "fineweb_tokenflow_baseline"
    load_stats = True

    # Model config
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    dim = 768
    n_heads = 12 # Head dim must be divisible by 128
    n_layers = 12
    tie_word_embeddings = True
    
    per_device_train_batch_size = 64
    gradient_accumulation_steps = 4
    
    dataloader_num_workers = 8
    max_steps = 30000
    eval_steps = 500
    save_steps = 1000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.98
    weight_decay = 0.02   
         
    warmup_steps = 2000
    lr_scheduler_type = "cosine"
    
    