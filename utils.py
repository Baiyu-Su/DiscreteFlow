import math
import torch
import torch.nn as nn

from typing import Tuple, Optional

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


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
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


def build_training_mask(M: int, N: int):
    """
    Returns a mask of shape (2*M*N, 2*M*N),
    where 0 => can attend, -inf => block.
    """
    total_seq = 2 * M * N
    mask = torch.full((total_seq, total_seq), float("-inf"), dtype=torch.bfloat16)

    row_ids = torch.arange(total_seq).unsqueeze(-1)
    col_ids = torch.arange(total_seq).unsqueeze(0)

    block_id_row = row_ids // N
    block_id_col = col_ids // N

    row_is_part1 = (block_id_row < M)
    col_is_part1 = (block_id_col < M)

    # Part1->Part1
    cond_p1p1 = row_is_part1 & col_is_part1 & (block_id_row >= block_id_col)
    mask[cond_p1p1] = 0.0

    # Part2->Part2
    cond_p2p2 = (~row_is_part1) & (~col_is_part1) & (block_id_row == block_id_col)
    mask[cond_p2p2] = 0.0

    # Part2->Part1
    cond_p2p1 = (~row_is_part1) & col_is_part1 & (block_id_row > (block_id_col + M))
    mask[cond_p2p1] = 0.0

    return mask


def build_inference_mask(start_pos: int, seq_len: int, N: int) -> torch.Tensor:
    """
    Build a block causal mask for inference.
    
    In this mask:
      - Both start_pos and seqlen must be multiples of N.
      - The tokens are conceptually split into blocks of size N.
      - Tokens within each block see each other (mask=0).
      - Across blocks, a standard causal mask is applied:
          For the i-th current block (0-indexed) the allowed blocks are all blocks
          with global index < (start_pos//N + i + 1).
    
    The final mask is built at the block level (shape: (num_current_blocks, total_blocks))
    and then expanded so that each scalar becomes an N*N sub-matrix.
    
    Args:
        start_pos (int): Number of cached tokens (a multiple of N).
        seq_len (int): Number of current tokens (a multiple of N).
        N (int): Block size.
        device: The torch device to allocate the mask.
        
    Returns:
        torch.Tensor: A mask tensor of shape (seqlen, start_pos + seqlen)
                      where allowed positions have value 0 and masked positions -inf.
    """
    num_cached_blocks = start_pos // N
    num_current_blocks = seq_len // N
    total_blocks = num_cached_blocks + num_current_blocks

    block_indices = torch.arange(total_blocks).unsqueeze(0)  # (1, total_blocks)
    allowed_limit = num_cached_blocks + torch.arange(1, num_current_blocks + 1).unsqueeze(1)  # (num_current_blocks, 1)
    block_mask = torch.where(
        block_indices < allowed_limit,
        torch.tensor(0.0),
        torch.tensor(float("-inf"))
    )
    ones_N = torch.ones((N, N))
    token_mask = torch.kron(block_mask, ones_N)
    return token_mask


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


def build_time_tensor(time: float, seq_len: int, B: int, N: int) -> torch.Tensor:
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
