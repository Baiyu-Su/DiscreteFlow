import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.functional as F

from typing import Tuple, Optional, Union


def init_normal(
    module: Union[nn.Linear, nn.Embedding],
    std: float,
    init_cutoff_factor: Optional[float] = None,
):
    """
    Initialize the weights of a module using a normal distribution.
    Args:
        module (Union[nn.Linear, nn.Embedding]): The module to initialize.
        std (float): The standard deviation of the normal distribution.
        init_cutoff_factor (Optional[float]): The cutoff factor for truncated normal distribution.
    Returns:
        None
    """
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)

    # biases
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def timestep_embedding(
    t: torch.Tensor,
    time_dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings that work for any shape of `t`.
    
    If `t` has shape (...), the result has shape (..., time_dim).
    Each scalar in `t` gets a separate sinusoidal embedding of size time_dim.
    
    Args:
        t (torch.Tensor): Arbitrary shape of float time values.
        time_dim (int): Dimension of the time embedding (output last dimension).
        max_period (int, optional): Maximum period for the sinusoidal functions.
        time_factor (float, optional): Scaling factor for time values.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (..., time_dim).
    """
    original_shape = t.shape
    t_flat = t.reshape(-1).float()  # [num_scalars]
    t_flat = t_flat * time_factor
    
    half = time_dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period) 
        * torch.arange(start=0, end=half, dtype=torch.float32, device=device) 
        / half
    )  # shape: [half]
    
    # Multiply each scalar in t_flat by each frequency => shape: (num_scalars, half)
    args = t_flat.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # shape: (num_scalars, 2*half)
    
    if time_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)  # shape: (num_scalars, time_dim)
    
    emb = emb.view(*original_shape, time_dim) # shape: original_shape + [time_dim]
    
    if torch.is_floating_point(t):
        emb = emb.to(t)
    
    return emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def build_training_block_mask(blk_num: int, blk_size: int, device=None):
    """
    Returns a *BlockMask* object (not a tensor) for FlexAttention.
    The pattern exactly matches your old dense mask logic.

    Part_1  (first  blk_num blocks)  : causal (lower-tri).
    Part_2  (second blk_num blocks) : block-diagonal.
    Part_2 → Part_1                 : allowed only if
                                      block_id_row > block_id_col + blk_num.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        block_id_row = q_idx // blk_size
        block_id_col = kv_idx // blk_size
        row_is_p1    = block_id_row < blk_num
        col_is_p1    = block_id_col < blk_num

        p1p1 = row_is_p1 & col_is_p1 & (block_id_row >= block_id_col)
        p2p2 = (~row_is_p1) & (~col_is_p1) & (block_id_row == block_id_col)
        p2p1 = (~row_is_p1) & col_is_p1 & (block_id_row > (block_id_col + blk_num))
        return p1p1 | p2p2 | p2p1

    total_tokens = 2 * blk_num * blk_size
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
        BLOCK_SIZE=blk_size,
        _compile=True,
    )
    

def build_inference_block_mask(max_blk_num: int, blk_size: int, device=None):
    """
    Block‑triangular *BlockMask* for inference‑time FlexAttention.

    Each block is completely visible to itself (bidirectional) **and** to
    every block that comes **before** it, but **never** to a *later* block.
    If only one block is present, the single block is therefore fully
    bidirectional, as requested.

    Args
    ----
    max_blk_num : int
        Maximum number of blocks that may appear on the KV side at inference
        (i.e. ⌈max_seq_len / blk_size⌉).  We build the mask once at this
        upper bound so it can be reused.
    blk_size    : int
        Number of tokens per block.
    device      : torch.device | str | None
        GPU/CPU on which the mask will live.  Pass the device of `q`.

    Returns
    -------
    BlockMask
        Ready to be fed to `flex_attention`.
    """

    #––– predicate evaluated by `create_block_mask` ––––––––––––––––––––––
    def mask_mod(b, h, q_idx, kv_idx):
        # Block indices for the query token and the key/value token
        q_blk = q_idx // blk_size
        kv_blk = kv_idx // blk_size
        # Allow iff the query's block is *not earlier* than the KV's block
        return q_blk >= kv_blk

    total_tokens = max_blk_num * blk_size        # square mask (Q = KV)

    return create_block_mask(
        mask_mod,
        B=None,                                  # batch‑/head‑agnostic
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
        BLOCK_SIZE=blk_size,
        _compile=True,
    )
    
    
def rectified_flow_interpolate(x0, x1, t):
    return t * x1 + (1 - t) * x0
    

def nucleus_cutoff(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) cutoff to a B x N x V tensor of probability distributions.
    
    For each pobability vector (of size V) along the last dimension, only the most probable tokens 
    whose cumulative probability mass is less than or equal to p are kept, and the rest are set to zero. 
    The surviving probabilities are then renormalized so that each vector sums to one.
    
    Args:
        probs (torch.Tensor): Tensor of shape (B, N, V) representing probability distributions.
        p (float): Cumulative probability threshold (0 < p <= 1).
        
    Returns:
        torch.Tensor: A tensor of the same shape as `probs` with filtered and renormalized probabilities.
    """
    probs_sorted, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # shape: (B, N, V)
    probs_cumsum = torch.cumsum(probs_sorted, dim=-1)
    mask = (probs_cumsum - probs_sorted) > p
    probs_sorted = probs_sorted.masked_fill(mask, 0.0)
    probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))
    probs_new = torch.empty_like(probs_sorted)
    probs_new.scatter_(-1, sorted_indices, probs_sorted)
    return probs_new


def build_inference_time(time: float, seq_len: int, B: int, N: int) -> torch.Tensor:
    """
    Create a time tensor based on the sequence length and time value.

    The input `seq_len` is assumed to be a multiple of N. Let k = seq_len // N.
    - If k == 1 (i.e. seq_len == N), the function returns a tensor of shape (B, 1)
      where each element is `time`.
    - If k > 1 (i.e. seq_len == k*N with k >= 2), the function returns a tensor of shape (B, k)
      where for each row the first k-1 elements are 1 and the last element is `time`.

    Args:
        seq_len (int): The total sequence length, a multiple of N.
        time (float): The time value to be used (for the single element in k==1 or the last element when k>1).
        batch_size (int): The batch size B.
        N (int): The factor such that seq_len is a multiple of N.
        device (str): The device to create the tensor on (default: "cuda").

    Returns:
        torch.Tensor: A tensor of shape (B, k) with the described values.
    """
    assert seq_len % N == 0, "seq_len must be an integer multiple of N"
    k = seq_len // N

    if k == 1:
        return torch.full((B, 1), time, dtype=torch.float)
    else:
        row = torch.cat([
            torch.ones(k - 1, dtype=torch.float) * 1.0,
            torch.tensor([time], dtype=torch.float)
        ])
        return row.unsqueeze(0).repeat(B, 1)
    

def row_rms(w: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute per‑row RMS with numerical epsilon."""
    return w.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()


def sample_gumbel(shape, beta=1, mu=0, eps=1e-20, device=None):
    """Sample from Gumbel(mu, beta)."""
    U = torch.rand(size=shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps) * beta + mu


def sample_posterior_gumbel_truncated(logits: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    """
    Sample Gumbel noise vectors from the posterior distribution P(G | argmax(L+G) = token_idx).
    
    This is a direct sampling method where one component is sampled from a truncated Gumbel.
    This does NOT guarantee the output is a standard Gumbel vector.

    Args:
        logits: Tensor of shape (B, T, V) - Teacher model logits
        token_idx: Tensor of shape (B, T) - Ground truth tokens
    
    Returns:
        Gumbel noise tensor of shape (B, T, V) satisfying the argmax constraint
    """
    # 1. Sample a full standard Gumbel vector
    gumbel_full = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    
    # 2. Find the maximum value among competitors (all tokens except the correct one)
    # Mask out the correct token by setting its logit to a very negative value
    masked_logits = logits.scatter(-1, token_idx.unsqueeze(-1), -1e9)
    max_others = (masked_logits + gumbel_full).max(dim=-1, keepdim=True).values
    
    # 3. Compute truncation point for the winning Gumbel
    # The winning token's Gumbel value must be > (max_others - logits_correct)
    truncation_point = max_others - logits.gather(-1, token_idx.unsqueeze(-1))
    
    # 4. Sample from truncated Gumbel distribution using inverse transform sampling
    # F(t) = exp(-exp(-t)) is the Gumbel CDF
    F_t = torch.exp(-torch.exp(-truncation_point))
    
    # Sample uniform from [F(t), 1] then apply inverse CDF
    U_t = F_t + torch.rand_like(F_t) * (1 - F_t)
    gumbel_x_new = -torch.log(-torch.log(U_t + 1e-20) + 1e-20)
    
    # 5. Replace the winning token's Gumbel value
    final_gumbel = gumbel_full.scatter(-1, token_idx.unsqueeze(-1), gumbel_x_new)
    
    return final_gumbel


def sample_posterior_gumbel(logits: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    """
    Sample from the posterior distribution P(z | argmax(z + logits) = token_idx) using the
    exact functional Gumbel-Max trick. This version is optimized for memory and speed.

    Args:
        logits: Tensor of shape (B, T, V) - Teacher model logits.
        token_idx: Tensor of shape (B, T) - Ground truth tokens.

    Returns:
        A Gumbel-like vector `z` of shape (B, T, V) satisfying argmax(z + logits) = token_idx.
    """
    # Reshape for processing
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    token_idx_flat = token_idx.reshape(B * T)
    
    log_p = F.log_softmax(logits_flat, dim=-1)
    
    batch_size, dim = log_p.shape # (B*T, V)

    # Sample Gumbel noise
    xi0 = sample_gumbel((batch_size, 1), device=logits.device)
    xii = sample_gumbel((batch_size, dim), device=logits.device)
    
    # Use logaddexp for numerical stability and memory efficiency
    # This computes log(exp(a) + exp(b)) which is more stable than doing it directly
    # Here, a = -xii and b = -xi0 + log_p
    post_gumbel_sample = -torch.logaddexp(-xii, -xi0 + log_p)
    
    # Get indices for in-place update
    batch_indices = torch.arange(batch_size, device=logits.device)
    
    # In-place update for the selected token using the formula
    post_gumbel_sample[batch_indices, token_idx_flat] = (xi0.squeeze(-1) - log_p[batch_indices, token_idx_flat])

    # Reshape back to original (B, T, V)
    return post_gumbel_sample.reshape(B, T, V)
