class MyConfig:
    ctx_len = 256

    output_dir = "/u/chizhang/scratch/data/out_shakespeare_gumbel"
    run_name = "shakespeare_tokenflow_gumbel"
    load_stats = True

    # Model config
    vocab_size = 32000  # LLaMA tokenizer vocab size
    dim = 512
    n_heads = 8
    n_layers = 8
    tie_word_embeddings = False
    
    # Gumbel reflow parameters
    use_gumbel_flow = True
    gumbel_conditioning_type = "cross_attention" # "x0" or "cross_attention"
    teacher_model_name = "/u/chizhang/scratch/data/out_shakespeare_teacher"  # Path to trained teacher
    
    per_device_train_batch_size = 64  
    gradient_accumulation_steps = 2 
    
    dataloader_num_workers = 4
    max_steps = 10000    
    eval_steps = 500
    save_steps = 1000
    logging_steps = 50

    # Optimization
    learning_rate = 4e-4 
    adam_beta2 = 0.98
    weight_decay = 0.02   
         
    warmup_steps = 500
    lr_scheduler_type = "cosine" 