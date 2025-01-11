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
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, rope_cache):
    seq_len = q.size(1)
    cos, sin = rope_cache
    cos = cos[:seq_len, None, :]
    sin = sin[:seq_len, None, :]
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_

def build_rope_cache(seq_len, head_dim, base=10000, device="cpu"):
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, freqs)  # (seq_len, half_dim//2)

    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (seq_len, half_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (seq_len, half_dim)
    return cos, sin

###########################################
# Using scaled_dot_product_attention
###########################################
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

    def forward(self, hidden_states, rope_cache, attn_mask=None):
        B, S, D = hidden_states.shape

        # 1) Project Q,K,V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # 2) Reshape => (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) RoPE
        if rope_cache is not None:
            q_t = q.transpose(1, 2)  # (B, S, h, hd)
            k_t = k.transpose(1, 2)  # (B, S, h, hd)
            q_t, k_t = apply_rope(q_t, k_t, rope_cache)
            q = q_t.transpose(1, 2)
            k = k_t.transpose(1, 2)

        # 4) Convert attn_mask=-inf => boolean mask for scaled_dot_product_attention
        if attn_mask is not None:
            bool_mask = (attn_mask == float("-inf"))  # shape (B, S, S)
            bool_mask = bool_mask.unsqueeze(1)        # shape (B, 1, S, S)
        else:
            bool_mask = None

        # 5) Built-in scaled_dot_product_attention
        # (B, num_heads, S, hd)
        attn_out, _ = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bool_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # 6) Reshape => (B, S, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)

        # 7) Final proj
        return self.out_proj(attn_out)

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

    def forward(self, hidden_states, rope_cache, attn_mask=None):
        # LN -> Self-Attn -> Residual
        hs_ln = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hs_ln, rope_cache=rope_cache, attn_mask=attn_mask)
        hidden_states = hidden_states + attn_out

        # LN -> MLP -> Residual
        hs_ln2 = self.post_attn_layernorm(hidden_states)
        ff_out = self.mlp(hs_ln2)
        hidden_states = hidden_states + ff_out

        return hidden_states

##########################################################
# Vectorized build_block_causal_mask
##########################################################
def build_block_causal_mask(M: int, N: int, device=torch.device("cpu")):
    """
    Returns a mask of shape (2*M*N, 2*M*N),
    where 0 => can attend, -inf => block.

    We define:
      - Part 1 blocks: block indices [0..M-1], each block has N tokens
      - Part 2 blocks: block indices [M..2*M-1], each block has N tokens

    The final 2D mask has size (2*M*N, 2*M*N).
    """
    total_seq = 2 * M * N
    mask = torch.full((total_seq, total_seq), float("-inf"), device=device, dtype=torch.float32)

    row_ids = torch.arange(total_seq, device=device).unsqueeze(-1)  
    col_ids = torch.arange(total_seq, device=device).unsqueeze(0)  

    block_id_row = row_ids // N  
    block_id_col = col_ids // N  

    row_is_part1 = (block_id_row < M)
    col_is_part1 = (block_id_col < M)

    # 1) Part1->Part1 (block-causal)
    cond_p1p1 = row_is_part1 & col_is_part1 & (block_id_row >= block_id_col)
    mask[cond_p1p1] = 0.0

    # 2) Part2->Part2 (only sees itself)
    cond_p2p2 = (~row_is_part1) & (~col_is_part1) & (block_id_row == block_id_col)
    mask[cond_p2p2] = 0.0

    # 3) Part2->Part1 (block i in part2 sees blocks 0..i in part1)
    # i.e. row_block_id >= col_block_id + M
    cond_p2p1 = (~row_is_part1) & col_is_part1 & (block_id_row >= (block_id_col + M))
    mask[cond_p2p1] = 0.0

    return mask

##########################################################
# The main discrete flow model
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

        # Precompute RoPE
        self.rope_cache = build_rope_cache(
            seq_len=self.total_seq,
            head_dim=(config.hidden_size // config.num_attention_heads),
            base=config.rope_scaling,
            device=torch.device("cpu")   # We'll store in CPU by default
        )

        # Precompute block-causal mask. Also on CPU initially.
        self.register_buffer(
            "block_causal_mask",
            build_block_causal_mask(M, N, device=torch.device("cpu")),
            persistent=False
        )

    def forward(self, input_embeddings: torch.Tensor, labels: torch.Tensor = None):
        B, seq_len, hidden_dim = input_embeddings.shape
        assert seq_len == self.total_seq, f"Expected seq_len={self.total_seq}, got {seq_len}"

        # 1) Move mask to same device as hidden_states
        attn_mask = self.block_causal_mask.to(input_embeddings.device)
        # shape => (2*M*N, 2*M*N); expand for batch dimension => (B, 2*M*N, 2*M*N)
        attn_mask = attn_mask.unsqueeze(0).expand(B, seq_len, seq_len)

        hidden_states = input_embeddings

        # 2) Pass through each block
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_cache=self.rope_cache, attn_mask=attn_mask)

        # 3) Final LN
        hidden_states = self.final_layer_norm(hidden_states)  # (B, 2*M*N, hidden_dim)

        # 4) Slice out Part2 => shape (B, M*N, hidden_dim)
        part2_states = hidden_states[:, self.M * self.N :, :]

        # 5) Project to logits => (B, M*N, vocab_size)
        logits_part2 = self.output_proj(part2_states)

        loss = None
        if labels is not None:
            # labels => shape (B, M*N)
            B_, length_ = labels.shape
            assert B_ == B and length_ == (self.M * self.N), \
                f"Labels must be shape (B, M*N). Got {labels.shape}"

            logits_2d = logits_part2.reshape(B * self.M * self.N, -1)
            labels_1d = labels.reshape(B * self.M * self.N)

            loss = F.cross_entropy(logits_2d, labels_1d)

        return {"loss": loss, "logits": logits_part2}
