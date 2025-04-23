class MyConfig:
    M = 64
    N = 16

    output_dir = "./out_small"
    run_name = "small_test_run"

    # Model config
    vocab_size = 32001
    dim = 768
    n_heads = 12
    n_layers = 12
    
    per_device_train_batch_size = 10
    gradient_accumulation_steps = 12
    
    dataloader_num_workers = 32
    warmup_steps = 1000
    max_steps = 16000
    eval_steps = 1600
    save_steps = 1600
    logging_steps = 10

    # Optimization
    learning_rate = 4e-4
    adam_beta2 = 0.99
    weight_decay = 0.02
    lr_scheduler_type = "cosine"
