class MyConfig:
    M = 128
    N = 8

    output_dir = "./out_med"
    run_name = "test_run"
    llama_checkpoint = "keeeeenw/MicroLlama"

    # Model config
    vocab_size = 32001
    dim = 1024
    n_heads = 16
    n_layers = 16
    
    per_device_train_batch_size = 24
    gradient_accumulation_steps = 5
    
    dataloader_num_workers = 4
    warmup_steps = 1000
    max_steps = 14000
    eval_steps = 2000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 6e-4
    adam_beta2 = 0.98
    weight_decay = 0.02
    lr_scheduler_type = "cosine"
