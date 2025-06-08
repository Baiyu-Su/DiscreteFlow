class MyConfig:
    blk_num = 8
    blk_size = 128

    output_dir = "./out_med"
    run_name = "med_run"

    # Model config
    vocab_size = 32000
    dim = 1024
    n_heads = 16
    n_layers = 16
    
    per_device_train_batch_size = 30
    gradient_accumulation_steps = 8
    
    dataloader_num_workers = 8
    max_steps = 40000
    eval_steps = 2000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    weight_decay = 0.02   
         
    warmup_steps = 2000
    lr_scheduler_type = "cosine"
    
    adam_beta2 = 0.98
