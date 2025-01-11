import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteFlowConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=12,
        max_sequence_length=1024,
        rope_scaling=10000,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_sequence_length = max_sequence_length
        self.rope_scaling = rope_scaling

def rotate_half(x):
    """
    Splits the last dimension in half and applies a rotation to create
    cos/sin embedding pairs.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, rope_cache):
    """
    q, k: (batch_size, seq_len, num_heads, head_dim)
    rope_cache: precomputed cos, sin (seq_len, head_dim)
    """
    # rope_cache might be shaped [seq_len, head_dim], containing cos/sin
    seq_len = q.size(1)
    cos, sin = rope_cache
    # cos, sin => shape (seq_len, head_dim) => broadcast to (1, seq_len, 1, head_dim)
    cos = cos[:seq_len, None, :]
    sin = sin[:seq_len, None, :]
    # q, k => (batch_size, seq_len, num_heads, head_dim)
    # we expand cos, sin => (seq_len, 1, head_dim)
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_

def build_rope_cache(seq_len, head_dim, base=10000, device="cpu"):
    """
    Precompute the cos/sin for rotary embedding.
    Typically used once at model init.
    """
    # half the dimension => head_dim//2 for cos/sin
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    # shape (half_dim/2,) => expand to build angles
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, freqs)  # (seq_len, half_dim//2)
    # cos, sin => (seq_len, half_dim)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin

class SelfAttention(nn.Module):
    """
    A standard multi-head self-attention using torch's built-in scaled_dot_product_attention,
    plus optional RoPE.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Q, K, V projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Final projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, rope_cache, attn_mask=None):
        """
        hidden_states: (B, seq_len, hidden_size)
        rope_cache:    (cos, sin) for RoPE, or None if not using.
        attn_mask:     (B, seq_len, seq_len) where 0 = can attend, -inf = block.
                       We'll adapt it to the scaled_dot_product_attention API.

        Returns:
          (B, seq_len, hidden_size)
        """
        B, S, D = hidden_states.shape
        # 1) Project to Q, K, V
        q = self.query(hidden_states)  # (B, S, D)
        k = self.key(hidden_states)    # (B, S, D)
        v = self.value(hidden_states)  # (B, S, D)

        # 2) Reshape to (B, S, num_heads, head_dim) then transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, S, hd)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, S, hd)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, S, hd)

        # 3) Apply RoPE (if using)
        if rope_cache is not None:
            # RoPE expects shape (B, S, num_heads, head_dim)
            # but we have (B, h, S, hd). Let's transpose back temporarily
            # or adapt apply_rope to handle (B, h, S, hd).
            # For clarity, we'll do a quick transpose -> apply -> transpose again.
            q_t = q.transpose(1, 2)  # (B, S, h, hd)
            k_t = k.transpose(1, 2)  # (B, S, h, hd)
            q_t, k_t = apply_rope(q_t, k_t, rope_cache)
            # now shape => (B, S, h, hd)
            q = q_t.transpose(1, 2)  # (B, h, S, hd)
            k = k_t.transpose(1, 2)  # (B, h, S, hd)

        # 4) Prepare the attention mask for scaled_dot_product_attention
        #    PyTorch expects shape (B, num_heads, S, S) or broadcastable.
        #    Right now we have (B, S, S). We'll unsqueeze dim=1 to broadcast over heads.
        if attn_mask is not None:
            # where mask == 0 => attend, -inf => block
            # scaled_dot_product_attention expects a float mask with +ve for block or 0 for keep
            # or a bool mask. We'll convert -inf => True (masked) and 0 => False
            bool_mask = (attn_mask == float("-inf"))  # shape (B, S, S)
            bool_mask = bool_mask.unsqueeze(1)        # shape (B, 1, S, S)
        else:
            bool_mask = None

        # 5) Call PyTorch's built-in scaled_dot_product_attention
        # q, k, v => (B, num_heads, S, head_dim)
        # mask => (B, 1, S, S) or (B, num_heads, S, S)
        # dropout_p => 0.0 for now
        # is_causal => False because we have an explicit mask
        attn_out, attn_weights = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bool_mask,
            dropout_p=0.0,
            is_causal=False
        )
        # attn_out => (B, num_heads, S, head_dim)

        # 6) Reshape back to (B, S, D)
        attn_out = attn_out.transpose(1, 2)  # (B, S, num_heads, head_dim)
        attn_out = attn_out.reshape(B, S, D)

        # 7) Final projection
        output = self.out_proj(attn_out)
        return output

class DiscreteFlowBlock(nn.Module):
    def __init__(self, config: DiscreteFlowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SelfAttention(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.post_attn_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),  # or SwiGLU, etc. LLaMA has a slightly different variant
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        )

    def forward(self, hidden_states, rope_cache, attn_mask=None):
        # Pre-attn LN
        hs_ln = self.input_layernorm(hidden_states)
        # Self-attn
        attn_out = self.self_attn(hs_ln, rope_cache, attn_mask=attn_mask)
        # Residual
        hidden_states = hidden_states + attn_out

        # Post-attn LN
        hs_ln2 = self.post_attn_layernorm(hidden_states)
        feedforward_out = self.mlp(hs_ln2)
        hidden_states = hidden_states + feedforward_out
        return hidden_states

def build_block_causal_mask(M: int, N: int, device: torch.device = torch.device("cpu")):
    """
    Returns a mask of shape (2*M*N, 2*M*N),
    where 0 => can attend, -inf => block.

    We define:
      - Part 1 blocks: block indices [0..M-1], each block has N tokens
      - Part 2 blocks: block indices [M..2*M-1], each block has N tokens

    The final 2D mask has size (2*M*N, 2*M*N). Index i => row, j => col.

    The logic is:
      1) Part1 -> Part1: block-causal. (Block i sees blocks 0..i fully)
      2) Part1 -> Part2: no visibility (all -inf)
      3) Part2 -> Part2: block i sees itself fully, no other block
      4) Part2 -> Part1: block i in part2 sees blocks 0..i in part1 fully
    """
    total_seq = 2 * M * N
    mask = torch.full((total_seq, total_seq), float("-inf"), device=device, dtype=torch.float32)

    # row_ids ranges [0..(total_seq-1)]^T, col_ids ranges [0..(total_seq-1)]
    row_ids = torch.arange(total_seq, device=device).unsqueeze(-1)  # shape (total_seq, 1)
    col_ids = torch.arange(total_seq, device=device).unsqueeze(0)   # shape (1, total_seq)

    # Identify block indices for each row/col
    # block_id_row = row_ids // N, block_id_col = col_ids // N
    block_id_row = row_ids // N  # shape (total_seq, 1)
    block_id_col = col_ids // N  # shape (1, total_seq)

    row_is_part1 = (block_id_row < M)
    col_is_part1 = (block_id_col < M)

    # Prepare to set mask=0.0 in places we "can attend".
    cond_p1p1 = row_is_part1 & col_is_part1 & (block_id_row >= block_id_col)
    mask[cond_p1p1] = 0.0

    cond_p2p2 = (~row_is_part1) & (~col_is_part1) & (block_id_row == block_id_col)
    mask[cond_p2p2] = 0.0

    cond_p2p1 = (~row_is_part1) & col_is_part1 & (block_id_row > (block_id_col + M))
    mask[cond_p2p1] = 0.0

    return mask

class FlowLlamaModel(nn.Module):
    def __init__(self, config: DiscreteFlowConfig, M=8, N=128):
        super().__init__()
        self.config = config
        self.M = M
        self.N = N
        self.total_seq = 2 * M * N  # total tokens in Part1+Part2

        # A stack of Llama blocks
        self.layers = nn.ModuleList(
            [DiscreteFlowBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

        # Output embedding not tied to input
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE cache
        # We pass (self.total_seq) as the maximum sequence length we might handle,
        # so we can do rotary for all positions. LLaMA approach might vary, but let's keep it simple.
        self.rope_cache = build_rope_cache(
            seq_len=self.total_seq,
            head_dim=(config.hidden_size // config.num_attention_heads),
            base=config.rope_scaling
        )

        # Build block-causal mask => (2*M*N, 2*M*N)
        # shape => (2*M*N, 2*M*N); we’ll expand for batch dimension in forward
        self.register_buffer(
            "block_causal_mask",
            build_block_causal_mask(M, N, device=torch.device("cpu")),
            persistent=False
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,  # (B, 2*M*N, hidden_dim)
        labels: torch.Tensor=None,       # (B, M*N) or (B, 2*M*N) but we only care about first M*N
    ):
        """
        Steps:
          1. Possibly apply LN to input or not.
          2. Create an expanded mask: shape (B, 2*M*N, 2*M*N).
          3. Pass hidden states through each block with the same mask.
          4. final_layer_norm -> output_proj
          5. Discard the Part 1 logits => keep Part 2 => compute cross-entropy with Part 1 labels.
        """
        B, seq_len, hidden_dim = input_embeddings.shape
        assert seq_len == self.total_seq, f"Expected seq_len={self.total_seq}, got {seq_len}"
        assert hidden_dim == self.config.hidden_size, f"Hidden dim mismatch"

        # We replicate the block_causal_mask for each batch,
        # shaping => (B, seq_len, seq_len).
        # Typically, we add dimension for "heads" if needed, or rely on broadcast.
        attn_mask = self.block_causal_mask.unsqueeze(0).expand(B, seq_len, seq_len)

        # Pass through each LlamaBlock
        hidden_states = input_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_cache=self.rope_cache, attn_mask=attn_mask)

        # Final LN
        hidden_states = self.final_layer_norm(hidden_states)  # (B, 2*M*N, hidden_size)

        # Compute logits for only the Part 2 chunk
        part2_states = hidden_states[:, self.M*self.N:, :]  # (B, M*N, hidden_size)
        logits_part2 = self.output_proj(part2_states)       # (B, M*N, vocab_size)

        # If we do training => we compare with the Part 1 tokens
        loss = None
        if labels is not None:
            # labels should be shape (B, M*N) containing the Part 1 token IDs.
            # We'll do standard cross-entropy => flatten (B*M*N, vocab_size) vs (B*M*N).
            B_, length_ = labels.shape
            assert B_ == B and length_ == (self.M*self.N), \
                f"Labels must be shape (B, M*N). Got {labels.shape} vs M*N={self.M*self.N}"

            logits_2d = logits_part2.reshape(B*self.M*self.N, -1)
            labels_1d = labels.reshape(B*self.M*self.N)
            loss = F.cross_entropy(logits_2d, labels_1d)

        return {
            "loss": loss,
            "logits": logits_part2  # shape (B, M*N, vocab_size)
        }


