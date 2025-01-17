import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig

##########################################################
# 1) DiscreteFlowConfig
##########################################################
class DiscreteFlowConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=12,
        max_sequence_length=1024,
        rope_scaling=10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_sequence_length = max_sequence_length
        self.rope_scaling = rope_scaling

##########################################################
# 2) RoPE Helpers
##########################################################
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    """
    q, k: shape (B, seq_len, num_heads, half_dim)
    cos, sin: shape (seq_len, half_dim) => on the same device as q/k

    We do:
       q_ = q*cos + rotate_half(q)*sin
       k_ = k*cos + rotate_half(k)*sin
    """
    # q, k => (B, S, h, half_dim)
    # cos, sin => (S, half_dim)
    B, S, h, half_dim = q.shape

    # expand cos, sin => (S, 1, half_dim) to broadcast over B,h
    cos = cos[:S].unsqueeze(1)  # (S, 1, half_dim)
    sin = sin[:S].unsqueeze(1)  # (S, 1, half_dim)

    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_

def build_rope_cache(seq_len, head_dim, base=10000):
    """
    We rotate half the dimension => half_dim = head_dim//2

    This returns (cos, sin) each shape (seq_len, head_dim//2).
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", t, freqs)  # shape (seq_len, half_dim//2)

    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (seq_len, half_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (seq_len, half_dim)
    return cos, sin

##########################################################
# 3) Self-Attention
##########################################################
class SelfAttention(nn.Module):
    def __init__(self, config: DiscreteFlowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, rope_cos, rope_sin, attn_mask=None):
        B, S, D = hidden_states.shape

        # 1) Project Q,K,V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # 2) Reshape => (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Split half-dim for RoPE
        half_dim = self.head_dim // 2
        q_rot, q_unrot = q.split(half_dim, dim=-1)
        k_rot, k_unrot = k.split(half_dim, dim=-1)

        q_rot_t = q_rot.transpose(1, 2)
        k_rot_t = k_rot.transpose(1, 2)

        # Apply RoPE
        q_rot_roped, k_rot_roped = apply_rope(q_rot_t, k_rot_t, rope_cos, rope_sin)

        q_rot = q_rot_roped.transpose(1, 2)
        k_rot = k_rot_roped.transpose(1, 2)

        q = torch.cat([q_rot, q_unrot], dim=-1)
        k = torch.cat([k_rot, k_unrot], dim=-1)

        # 4) Convert attn_mask=-inf => boolean mask
        if attn_mask is not None:
            bool_mask = (attn_mask == float("-inf"))
            bool_mask = bool_mask.unsqueeze(1)  # (B, 1, S, S)
        else:
            bool_mask = None

        # 5) Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bool_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # 6) Reshape => (B, S, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)

        # 7) Final projection
        return self.out_proj(attn_out)


##########################################################
# 4) DiscreteFlowBlock
##########################################################
class DiscreteFlowBlock(nn.Module):
    def __init__(self, config: DiscreteFlowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SelfAttention(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.post_attn_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def forward(self, hidden_states, rope_cos, rope_sin, attn_mask=None):
        # LN -> Self-Attn -> Residual
        hs_ln = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hs_ln, rope_cos, rope_sin, attn_mask=attn_mask)
        hidden_states = hidden_states + attn_out

        # LN -> MLP -> Residual
        hs_ln2 = self.post_attn_layernorm(hidden_states)
        ff_out = self.mlp(hs_ln2)
        hidden_states = hidden_states + ff_out

        return hidden_states

##########################################################
# 5) build_block_causal_mask
##########################################################
def build_block_causal_mask(M: int, N: int):
    """
    Returns a mask of shape (2*M*N, 2*M*N),
    where 0 => can attend, -inf => block.
    """
    total_seq = 2 * M * N
    mask = torch.full((total_seq, total_seq), float("-inf"), dtype=torch.float32)

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
    cond_p2p1 = (~row_is_part1) & col_is_part1 & (block_id_row >= (block_id_col + M))
    mask[cond_p2p1] = 0.0

    return mask

##########################################################
# 6) DiscreteFlowModel
##########################################################
class DiscreteFlowModel(nn.Module):
    def __init__(self, config: DiscreteFlowConfig, M=8, N=128):
        super().__init__()
        self.config = config
        self.M = M
        self.N = N
        self.total_seq = 2 * M * N

        self.layers = nn.ModuleList(
            [DiscreteFlowBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 1) Precompute RoPE on CPU, then store as buffers
        cos, sin = build_rope_cache(
            seq_len=self.total_seq,
            head_dim=(config.hidden_size // config.num_attention_heads),
            base=config.rope_scaling,
        )
        # Register them as buffers so they move with model.to(device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # 2) Build block-causal mask => also store as buffer
        mask = build_block_causal_mask(M, N)
        self.register_buffer("block_causal_mask", mask, persistent=False)

    def forward(self, input_embeddings: torch.Tensor, labels: torch.Tensor = None):
        """
        input_embeddings: shape (B, 2*M*N, hidden_size)
        labels: shape (B, M*N) => IDs for part1
        """
        B, seq_len, hidden_dim = input_embeddings.shape
        assert seq_len == self.total_seq, f"Expected seq_len={self.total_seq}, got {seq_len}"

        # 1) Expand attn_mask => (B, seq_len, seq_len) on correct device
        attn_mask = self.block_causal_mask  # already a buffer
        attn_mask = attn_mask.unsqueeze(0).expand(B, seq_len, seq_len).to(input_embeddings.device)

        # 2) Also ensure rope_cos, rope_sin are on the same device
        rope_cos = self.rope_cos.to(input_embeddings.device)
        rope_sin = self.rope_sin.to(input_embeddings.device)

        hidden_states = input_embeddings

        # 3) Pass through each block
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attn_mask=attn_mask
            )

        # 4) Final LN
        hidden_states = self.final_layer_norm(hidden_states)  # (B, 2*M*N, hidden_size)

        # 5) Slice out Part2 => shape (B, M*N, hidden_dim)
        part2_states = hidden_states[:, self.M * self.N :, :]

        # 6) Project => (B, M*N, vocab_size)
        logits_part2 = self.output_proj(part2_states)

        # 7) Loss
        loss = None
        if labels is not None:
            B_, length_ = labels.shape
            assert B_ == B and length_ == (self.M * self.N), \
                f"Labels must be shape (B, M*N). Got {labels.shape}"

            logits_2d = logits_part2.reshape(B * self.M * self.N, -1)
            labels_1d = labels.reshape(B * self.M * self.N)

            loss = F.cross_entropy(logits_2d, labels_1d)

        return {"loss": loss, "logits": logits_part2}
