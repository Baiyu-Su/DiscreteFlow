class MyConfig:
    ctx_len = 256

    output_dir = "/u/chizhang/scratch/data/out_shakespeare_teacher"
    run_name = "shakespeare_llama_teacher"

    # Model config - same architecture as TokenFlow for fair comparison
    vocab_size = 32000  # LLaMA tokenizer vocab size
    dim = 512           # hidden_size in LLaMA config
    hidden_dim = 2048   # intermediate_size in LLaMA config (4 * dim)
    n_heads = 8
    n_kv_heads = 8      # Same as n_heads for standard attention
    n_layers = 8
    norm_eps = 1e-6     # rms_norm_eps in LLaMA config
    tie_word_embeddings = True
    
    per_device_train_batch_size = 128  
    gradient_accumulation_steps = 1  
    
    dataloader_num_workers = 4
    max_steps = 5000    
    eval_steps = 100 # Keep frequent because model tends to overfit quickly
    save_steps = 1000
    logging_steps = 50

    # Optimization
    learning_rate = 4e-4 
    adam_beta2 = 0.98
    weight_decay = 0.02   
         
    warmup_steps = 500   # Reduced since fewer total steps
    lr_scheduler_type = "cosine" 