import os
import json
import torch
import argparse

from safetensors.torch import load_file

import numpy as np
from scipy.stats import beta
from transformers import LlamaTokenizer
from model import TokenFlowModel, TokenFlowConfig

# torchrun --standalone examine.py --model_dir out_med --prompt_file prompts.json

def load_model(model_dir, device):
    """
    Loads the saved model weights from model.safetensors and creates
    a TokenFlowModel with the same hyperparameters used during training.
    """

    # 1) Load state_dict from "model.safetensors"
    model_weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Could not find {model_weights_path}")

    state_dict = load_file(model_weights_path)

    # 2) Manually specify (or otherwise retrieve) the config used during training:
    #    Replace these values with the ones you actually used for training.
    #    For example, if you used M=4, N=256, dim=1024, etc., put those exact numbers here.
    model_config = TokenFlowConfig(
        is_inference=True,  # we want inference mode
        M=128,                # <-- change to match your training
        N=8,              # <-- change to match your training
        vocab_size=32001,   # same as you used in training
        dim=1024,           # ...
        n_heads=16,
        n_layers=16,
    )

    # 3) Create the model and load the weights
    model = TokenFlowModel(model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Examine the generation quality of the trained TokenFlow model.")
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Directory containing the saved model.safetensors and trainer_state.json.")
    parser.add_argument("--prompt_file", type=str, default="prompts.json", 
                        help="Path to the JSON file with text prompts.")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: GPU not available. The generate function may assume CUDA and could error.")

    # Load the tokenizer (adjust the path to your local Llama tokenizer)
    tokenizer = LlamaTokenizer.from_pretrained("./.hf_llama")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    bos_id = tokenizer.bos_token_id

    # Load the custom TokenFlow model
    model = load_model(args.model_dir, device)

    # Load prompts from the JSON file
    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list):
        raise ValueError("The JSON file must contain a list of prompt strings.")

    # Tokenize each prompt
    raw_tokens = [
        tokenizer.encode(p, add_special_tokens=False)
        for p in prompts
    ]

    # 3) Prepend BOS and collect lengths
    prompt_tokens = [
        [bos_id] + toks
        for toks in raw_tokens
    ]

    # invert the Beta(2,6) CDF at those levels
    time_schedule = np.linspace(0, 1., 128)

    print(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")

    # Generate completions (assuming your TokenFlowModel implements .generate())
    completions_tokens = model.generate(
        prompt_tokens,
        time_schedule=time_schedule,
        top_p=0.95,
        echo=False,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
    )

    # Decode the generated tokens
    # completions = [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in completions_tokens]
    completions = []
    for tokens in completions_tokens:
        completion = tokenizer.decode(tokens, skip_special_tokens=False)
        completions.append(completion)

    # Print results
    for prompt, completion in zip(prompts, completions):
        print("Prompt:")
        print(prompt)
        print("\nCompletion:")
        print(completion)
        print("-" * 80)


if __name__ == "__main__":
    main()
