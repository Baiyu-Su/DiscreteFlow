import torch
import matplotlib.pyplot as plt
from model import build_block_causal_mask

def main():
    M = 4
    N = 6
    mask = build_block_causal_mask(M, N, device=torch.device("cpu"))
    print("Mask shape:", mask.shape)
    print("Sample mask values:\n", mask)

    # Let's visualize it
    plt.figure(figsize=(6,6))
    # We'll convert -inf to something like -10 for display
    display_mask = mask.clone()
    display_mask[display_mask == float("-inf")] = -10.0

    plt.imshow(display_mask, cmap="viridis")
    plt.title(f"Block-Causal Mask (M={M}, N={N})")
    plt.colorbar(label="Attention Weight")
    plt.savefig("mask_visual.png")
    plt.show()

if __name__ == "__main__":
    main()
