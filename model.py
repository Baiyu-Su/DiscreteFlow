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

def build_block_causal_mask(
    M: int, 
    N: int, 
    device: torch.device = torch.device("cpu")
):
    """
    Returns a mask of shape (2*M*N, 2*M*N)
    where 0 => can attend, -inf => block.

    Indices:
      Part 1 block i: i*N .. i*N + (N-1)
      Part 2 block i: (M + i)*N .. (M + i)*N + (N-1)

    For each query index (row), we set -inf to those we cannot attend to.

    We'll do it in block form for clarity.
    """
    total_seq = 2 * M * N
    mask = torch.zeros((total_seq, total_seq), dtype=torch.float32, device=device)

    # Helper to get the start index of block i in part 1 or part 2
    def block_start(part, i):
        # part: 1 or 2
        # i: block index in [0..M-1]
        if part == 1:
            return i * N
        else:  # part == 2
            return (M + i) * N

    # PART 1 <-> PART 1 block-causal
    # For block i, it can see blocks 0..i fully.
    for i in range(M):
        start_i = block_start(1, i)
        end_i = start_i + N  # exclusive
        for j in range(i+1):  # blocks 0..i
            start_j = block_start(1, j)
            end_j = start_j + N
            # allow full attend
            mask[start_i:end_i, start_j:end_j] = 0.0  # can attend

        # But block i cannot see block (j) > i
        # so set them to -inf
        for j in range(i+1, M):
            start_j = block_start(1, j)
            end_j = start_j + N
            mask[start_i:end_i, start_j:end_j] = float("-inf")

    # PART 2 <-> PART 2: block i can only see itself
    for i in range(M):
        start_i_2 = block_start(2, i)
        end_i_2 = start_i_2 + N
        # block i sees itself
        mask[start_i_2:end_i_2, start_i_2:end_i_2] = 0.0
        # other blocks => -inf
        for j in range(M):
            if j != i:
                start_j_2 = block_start(2, j)
                end_j_2 = start_j_2 + N
                mask[start_i_2:end_i_2, start_j_2:end_j_2] = float("-inf")

    # PART 2 -> PART 1: block i in part 2 can see blocks [0..i] in part 1
    # so if i2 is the i-th block in part 2, then it can see blocks 0..i in part 1
    for i in range(M):
        start_i_2 = block_start(2, i)
        end_i_2 = start_i_2 + N
        for j in range(i+1):  # block 0..i in part 1
            start_j_1 = block_start(1, j)
            end_j_1 = start_j_1 + N
            mask[start_i_2:end_i_2, start_j_1:end_j_1] = 0.0

        # But blocks > i are not visible
        for j in range(i+1, M):
            start_j_1 = block_start(1, j)
            end_j_1 = start_j_1 + N
            mask[start_i_2:end_i_2, start_j_1:end_j_1] = float("-inf")

    # PART 1 -> PART 2: Part 1 tokens see NO Part 2 tokens
    for i in range(M):
        start_i_1 = block_start(1, i)
        end_i_1 = start_i_1 + N
        for j in range(M):
            start_j_2 = block_start(2, j)
            end_j_2 = start_j_2 + N
            mask[start_i_1:end_i_1, start_j_2:end_j_2] = float("-inf")

    return mask  # shape (2*M*N, 2*M*N)

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


