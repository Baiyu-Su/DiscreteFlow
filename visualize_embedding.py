import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel
from scipy.special import gammaln

def expected_l2_norm(d):
    """
    Compute E[||X||_2] for X ~ N(0, I_d) in a numerically stable way.
    
    Args:
      d (int or array_like): Dimension(s), must be > 0.
    
    Returns:
      float or ndarray: The mean L2 norm.
    """
    # ensure float array
    d = np.asarray(d, dtype=np.float64)
    # log of gamma((d+1)/2) minus log of gamma(d/2)
    log_ratio = gammaln((d + 1) * 0.5) - gammaln(d * 0.5)
    return np.sqrt(2.0) * np.exp(log_ratio)


# Define your checkpoint path. Replace with the actual checkpoint location.
checkpoint = "keeeeenw/MicroLlama"

# Load the model
model = AutoModel.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    device_map="cpu"  # load on CPU to avoid GPU memory spike
)

# Get the pretrained token embedding from the model.
pretrained_token_embedding = model.get_input_embeddings()

# Extract the updated embedding weights as a NumPy array.
# Assuming the shape is (32000, d) where d is the embedding dimension.
embedding_weights = pretrained_token_embedding.weight.data.cpu().numpy()

# Compute the L2 norm for each embedding vector (each row in the embedding matrix).
l2_norms = np.linalg.norm(embedding_weights, axis=1)

d = float(embedding_weights.shape[1])

median_norm = np.median(l2_norms) / expected_l2_norm(d)
print(f"Median embedding norm: {median_norm:.4f}")

# Compute the threshold value: 0.02 * sqrt(d)
threshold_value = 0.04 * expected_l2_norm(d)

# Create the histogram of L2 norms.
plt.figure(figsize=(8, 5))
plt.hist(l2_norms, bins=50, alpha=0.7, color='blue')
plt.xlabel("L2 Norm of Embedding Vectors")
plt.ylabel("Frequency")
plt.title("Distribution of L2 Norms of Embedding Vectors")

# Add a vertical line at the threshold value.
plt.axvline(x=threshold_value, color='red', label="$\sqrt{d}$")
plt.legend()

# Save the figure to a file instead of displaying it.
plt.savefig("embedding_norm_distribution.png")
plt.close()

