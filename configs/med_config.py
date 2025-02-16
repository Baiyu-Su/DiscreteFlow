class MyConfig:
    M = 8
    N = 128

    output_dir = "./out"
    run_name = "test_run"

    # Model config
    vocab_size = 32000
    dim = 1024
    n_heads = 16
    n_layers = 12
    
    per_device_train_batch_size = 30
    gradient_accumulation_steps = 4
    
    dataloader_num_workers = 16
    warmup_steps = 1000
    max_steps = 20000
    eval_steps = 1000
    save_steps = 2000
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.95
    lr_scheduler_type = "cosine"
