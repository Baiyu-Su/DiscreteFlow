from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from torch import Tensor

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from typing import Tuple, Optional, Union

from model_utils import (
    init_normal,
    precompute_freqs_cis,
    apply_rotary_emb,
    build_training_block_mask,
    build_inference_block_mask,
    timestep_embedding,
    rectified_flow_interpolate,
    build_inference_time,
)


class TokenFlowConfig(PretrainedConfig):
    model_type = "tokenflow"  # used internally by HF for registry & save/load

    def __init__(
        self,
        blk_num: int = 128,
        blk_size: int = 8,
        max_batch: int = 64,
        vocab_size: int = 32000,
        dim: int = 1024,
        hidden_dim: int | None = None,
        time_dim: int = 256,
        n_heads: int = 16,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 12,
        init_cutoff_factor: float = 3.0,
        embed_scale: float | None = None,
        tie_word_embeddings = True,
        rope_scaling: int = 10000,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        load_stats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blk_num = blk_num
        self.blk_size = blk_size
        self.max_batch = max_batch
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.time_dim = time_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_layers = n_layers
        self.init_cutoff_factor = init_cutoff_factor
        self.embed_scale = embed_scale or 1.0 / math.sqrt(dim)
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_scaling = rope_scaling
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.load_stats = load_stats


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class ModulationOut:
    """
    Output of the adaLN Modulation layer.
    Attributes:
        shift (torch.Tensor): Shift parameter for modulation.
        scale (torch.Tensor): Scale parameter for modulation.
        gate (torch.Tensor): Gate parameter for modulation.
    """
    shift: torch.Tensor  # shape: (B, M, dim)
    scale: torch.Tensor  # shape: (B, M, dim)
    gate:  torch.Tensor  # shape: (B, M, dim)


class Modulation(nn.Module):
    def __init__(self, config: TokenFlowConfig):
        """
        Initialize the adaLN Modulation layer.
        Args:
            dim (int): Model dimension.
        Attributes:
            w (nn.Linear): Linear transformation for modulation (3 terms).
        """
        super().__init__()
        self.dim = config.dim
        self.init_cutoff_factor = config.init_cutoff_factor
        self.w = nn.Linear(self.dim, 3 * self.dim)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.dim)
        init_normal(self.w, std, self.init_cutoff_factor)

    def forward(self, vec: torch.Tensor) -> ModulationOut:
        """
        Apply modulation to the input tensor.
        Args:
            vec(torch.Tensor): (B, M, dim) — one vector per block.
        Returns:
            ModulationOut: modulation parameters (shift, scale, gate) each of shape (B, M, dim)
        """
        out = self.w(F.silu(vec))  # (B, M, 3*dim)
        shift, scale, gate = out.chunk(3, dim=-1)  # each (B, M, dim)
        return ModulationOut(shift=shift, scale=scale, gate=gate)


class MLPEmbedder(nn.Module):
    def __init__(self, config: TokenFlowConfig):
        """
        Initialize the MLPEmbedder module.
        Args:
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            in_dim (int): Input dimension of time vector.
            dim (int): Model dimension.
            init_cutoff_factor (float): Cutoff factor for weight initialization.
            wi (nn.Linear): Linear transformation for input.
            silu (nn.SiLU): Swish activation function.
            wo (nn.Linear): Linear transformation for output.
        """
        super().__init__()
        self.in_dim = config.time_dim
        self.dim = config.dim
        self.init_cutoff_factor = config.init_cutoff_factor
        self.wi = nn.Linear(self.in_dim, self.dim, bias=True)
        self.silu = nn.SiLU()
        self.wo = nn.Linear(self.dim, self.dim, bias=True)

    def reset_parameters(self):
        in_std = 1 / math.sqrt(self.in_dim)
        out_std = 1 / math.sqrt(self.dim)
        init_normal(self.wi, in_std, self.init_cutoff_factor)
        init_normal(self.wo, out_std, self.init_cutoff_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPEmbedder module.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying linear transformation and activation.
        """
        return self.wo(self.silu(self.wi(x)))


class Attention(nn.Module):
    """
    Multi-head attention w/ rotary positional encodings + FlexAttention.
    Args:
        config (TokenFlowConfig): Model configuration parameters.
    Attributes:
        n_kv_heads (int): Number of key/value heads.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension size of each attention head.
        blk_size (int): Block size.
        blk_num (int): Number of blocks.
        wq (nn.Linear): Linear transformation for query.
        wk (nn.Linear): Linear transformation for key.
        wv (nn.Linear): Linear transformation for value.
        wo (nn.Linear): Linear transformation for output.
        _flex_attn (function): FlexAttention function.
    """

    def __init__(self, layer_id: int, config: TokenFlowConfig):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.blk_size = config.blk_size
        self.blk_num = config.blk_num
        self.init_cutoff_factor = config.init_cutoff_factor

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        if config.blk_size >= 128:
            self._flex_attn = lambda q, k, v, block_mask, scale, enable_gqa: flex_attention(
                q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa
            )
        else:
            _tile_sz = config.blk_size

            def _flex_small(q, k, v, block_mask, scale, enable_gqa, opts):
                return flex_attention(
                    q, k, v,
                    block_mask=block_mask,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    kernel_options=opts,
                )
            _compiled_flex_small = torch.compile(
                _flex_small,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs"
            )
            _kernel_opts = {
                "BLOCK_M":  _tile_sz, "BLOCK_N":  _tile_sz,
                "BLOCK_M1": _tile_sz, "BLOCK_N1": _tile_sz,
                "BLOCK_M2": _tile_sz, "BLOCK_N2": _tile_sz,
            }

            self._flex_attn = functools.partial(_compiled_flex_small, opts=_kernel_opts)

        self._block_mask_cache: dict[torch.device, "BlockMask"] = {}
        
        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.dim)
        attn_out_std = 1 / (math.sqrt(2 * self.dim * (self.layer_id + 1)))

        init_normal(self.wq, std, self.init_cutoff_factor)
        init_normal(self.wk, std, self.init_cutoff_factor)
        init_normal(self.wv, std, self.init_cutoff_factor)
        init_normal(self.wo, attn_out_std, self.init_cutoff_factor)

    def forward(self, x: Tensor, start_pos: int, freqs_cis, use_cache=False):
        """
        Args
        ----
        x          : (B, L, D)
        start_pos  : starting token position in the *full* sequence
                     (needed only for KV-cache path)
        freqs_cis  : rotary embedding tensor
        _unused_mask : kept for API backward-compat (ignored - masking handled
                       internally by FlexAttention)
        """
        batch_size, seq_len, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # (B, L, H, D_h)
        q = q.view(batch_size, seq_len, self.n_heads,    self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if not use_cache:
            q = q.transpose(1, 2)                    # (B, H_q, L, D_h)
            k = k.transpose(1, 2)                    # (B, H_kv,L, D_h)
            v = v.transpose(1, 2)
            
            dev = q.device                                     # q/k/v's GPU
            if dev not in self._block_mask_cache:              # NEW
                self._block_mask_cache[dev] = build_training_block_mask(
                    self.blk_num, self.blk_size, device=dev
                )
            block_mask = self._block_mask_cache[dev] 

            out = self._flex_attn(
                q, k, v,
                block_mask=block_mask,    # block‑sparse causal mask
                scale=1.0 / math.sqrt(self.head_dim),
                enable_gqa=(self.n_kv_heads != self.n_heads),
                opts=self._flex_attn_opts
            )        
        else:
            end_pos = start_pos + seq_len
            
            if self.cache_k.numel() == 0 or self.cache_k.size(1) < end_pos:
                max_len = self.blk_num * self.blk_size
                shape   = (self.blk_size, max_len, self.n_kv_heads, self.head_dim)
                self.cache_k = torch.zeros(shape, device=x.device, dtype=k.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                
            self.cache_k[:batch_size, start_pos:end_pos] = k
            self.cache_v[:batch_size, start_pos:end_pos] = v

            k_full = self.cache_k[:batch_size, :end_pos]      # (B, T, H_kv, D_h)
            v_full = self.cache_v[:batch_size, :end_pos]

            q = q.transpose(1, 2)                    # (B, H_q, L,  D_h)
            k = k_full.transpose(1, 2)               # (B, H_kv,T, D_h)
            v = v_full.transpose(1, 2)
            
            dev = q.device                                     # q/k/v's GPU
            if dev not in self._block_mask_cache:              # NEW
                self._block_mask_cache[dev] = build_inference_block_mask(
                    self.blk_num, self.blk_size, device=dev
                )
            block_mask = self._block_mask_cache[dev] 

            out = self._flex_attn(
                q, k, v,
                block_mask=block_mask._adjust(q.shape[-2], k.shape[-2]),                     # causal via cache
                scale=1.0 / math.sqrt(self.head_dim),
                enable_gqa=(self.n_kv_heads != self.n_heads),
            )

        # (B, L, H, D_h) → (B, L, D)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, layer_id: int, config: TokenFlowConfig):
        """
        Initialize the FeedForward module.
        Args:
            layer_id (int): Identifier for the layer.
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            w1 (nn.Linear): Linear transformation for the first layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third layer.
        """
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        self.multiple_of = config.multiple_of
        self.init_cutoff_factor = config.init_cutoff_factor

        hidden_dim = self.multiple_of * ((self.dim + self.multiple_of - 1) // self.multiple_of)

        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.dim)
        ff_out_std = 1 / (math.sqrt(2 * self.dim * (self.layer_id + 1)))
        init_normal(self.w1, std, self.init_cutoff_factor)
        init_normal(self.w2, std, self.init_cutoff_factor)
        init_normal(self.w3, ff_out_std, self.init_cutoff_factor)

    def forward(self, x):
        """
        Forward pass of the FeedForward module.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying feedforward layers.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class NormalizedEmbedding(nn.Module):
    """Embedding layer whose vectors are normalized to a target RMS value.

    The normalization is performed in-place within the forward pass, ensuring
    that the stored weights are always normalized. This operation is done
    without tracking gradients.

    Attributes:
        raw_weight (nn.Parameter): The learnable parameter tensor.
        scale (float): The target RMS value for the embedding vectors.
    """

    def __init__(self, config: TokenFlowConfig):
        super().__init__()
        
        if not isinstance(config.embed_scale, float) or config.embed_scale <= 0.0:
            raise ValueError("The `embed_scale` must be a positive float.")
            
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.embed_scale = config.embed_scale
        self.padding_idx = None
        self.scale_grad_by_freq = False
        self.sparse = False
        self.eps = config.norm_eps

        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes and normalizes the embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """
        Performs in-place RMS normalization of the weight tensor.
        This method is designed to be called within a `torch.no_grad()` context.
        """
        self.weight.data = F.normalize(self.weight.data, p=2, dim=1) * self.embed_scale * math.sqrt(self.dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass with in-place weight normalization.
        """
        with torch.no_grad():
            self._normalize_weights()
            
        return F.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        
    def extra_repr(self) -> str:
        """Adds extra information to the module representation."""
        s = f'{self.vocab_size}, {self.dim}'
        if self.embed_scale != 1.0:
            s += f', scale={self.embed_scale}'
        return s
    

class TokenFlowBlock(nn.Module):
    def __init__(self, layer_id: int, config: TokenFlowConfig):
        """
        Initialize a TokenFlowBlock.
        Args:
            layer_id (int): Identifier for the layer.
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            blk_size (int): Block size.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            modulation (Modulation): adaLN modulation module.
        """
        super().__init__()
        self.blk_size = config.blk_size
        self.attention = Attention(layer_id, config)
        self.feed_forward = FeedForward(layer_id, config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.modulation = Modulation(config)

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.modulation.reset_parameters()
    
    def forward(
        self, 
        x: torch.Tensor, 
        vec: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        use_cache: bool = False
    ) -> torch.Tensor:
        mod = self.modulation(vec) # (B, 2*M, dim)

        batch_size, ctx = x.shape[0], x.shape[1]
        x_blk      = x.view(batch_size, -1, self.blk_size, self.dim) # (B, 2*M, N, D)
        mod_shift  = mod.shift.unsqueeze(2)  # (B, 2*M, 1, D)
        mod_scale  = mod.scale.unsqueeze(2)
        mod_gate   = mod.gate .unsqueeze(2)

        x_mod = (1 + mod_scale) * self.attention_norm(x_blk) + mod_shift
        x_mod_flat = x_mod.view(batch_size, ctx, self.dim)                  # (B, ctx, D)
        h = x + self.attention(x_mod_flat, start_pos, freqs_cis, use_cache)

        ffn_out_blk = self.feed_forward(self.ffn_norm(h)).view(
            batch_size, -1, self.blk_size, self.dim
        ) # (B, 2*M, N, D)

        h_blk = h.view(batch_size, -1, self.blk_size, self.dim)
        out_blk = h_blk + mod_gate * ffn_out_blk # broadcast over N

        return out_blk.view(batch_size, ctx, self.dim)


class TokenFlowModel(PreTrainedModel):
    def __init__(self, config: TokenFlowConfig):
        """
        Initialize a Token Flow Model.
        Args:
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            config (TokenFlowConfig): Model configuration parameters.
            load_stats (bool): Whether to load statistics.
            n_layers (int): Number of layers in the model.
            n_heads (int): Number of attention heads.
            blk_num (int): Number of blocks.
            blk_size (int): Block size.
            max_batch (int): Maximum batch size for inference.
            dim (int): Model dimension.
            time_dim (int): Dimension of the time vector.
            token_embed (nn.Embedding): Token embedding layer.
            time_embed (MLPEmbedder): Time embedding layer.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            final_layer_norm (RMSNorm): Final layer normalization.
            output_proj (nn.Linear): Output projection layer.
            freqs_cis (torch.Tensor): Precomputed frequency tensors.
        """
        super().__init__(config)
        self.config = config
        self.load_stats = config.load_stats
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.blk_num = config.blk_num
        self.blk_size = config.blk_size
        self.max_batch = config.max_batch
        self.dim = config.dim
        self.time_dim = config.time_dim
        self.embed_scale = config.embed_scale
        self.tied_word_embeddings = config.tie_word_embeddings

        self.token_embed = NormalizedEmbedding(config)
        # self.beta_dist = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(5.0))
        self.time_embed = MLPEmbedder(config)

        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TokenFlowBlock(layer_id, config))

        if config.tie_word_embeddings:
            self.final_layer_norm = nn.LayerNorm(
                config.dim,
                eps=config.norm_eps,
                elementwise_affine=True
            )
        else:
            self.final_layer_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.output_proj = nn.Linear(config.dim, config.vocab_size, bias=False)
            
        freqs_cis = precompute_freqs_cis(
            self.dim//self.n_heads, self.blk_num*self.blk_size, config.rope_scaling)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    
    def _rope_cache(self, device):
        if (not hasattr(self, "_freqs_cis") or self._freqs_cis.device != device):
            freqs_cis = precompute_freqs_cis(
                self.dim // self.n_heads,
                self.blk_num * self.blk_size,
                self.config.rope_scaling,
            ).to(device)
            self.register_buffer("_freqs_cis", freqs_cis, persistent=False)
        return self._freqs_cis
    
    
    # @torch._dynamo.disable
    # def _sample_time(self, shape, device):
    #     return self.beta_dist.sample(shape).to(device)
    
        
    def _compute_singular_logits(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        std: float = 1.0, 
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Args:
            xt (torch.Tensor): (B, seq_len=M*N, d)
            t (torch.Tensor): (B, seq_len)           -- per-position time
            std (float): Standard deviation of the noise
        Returns:
            torch.Tensor: flow_logits of shape (B, seq_len, V)
        """
        token_embeddings = self.token_embed.weight

        xt_norm2 = xt.pow(2).sum(dim=-1, keepdim=True) # (B, seq_len, 1)
        e_norm2 = token_embeddings.pow(2).sum(dim=-1).unsqueeze(0).unsqueeze(0)

        cross = xt @ token_embeddings.t() # (B, seq_len, V)
        numer = xt_norm2 - 2 * t * cross + (t**2) * e_norm2  # (B, seq_len, V)
        denom = (2 * std ** 2) * (1 - t).pow(2) + eps ** 2 # (B, seq_len, 1)
        logits = -numer / denom # (B, seq_len, V)

        logZ = torch.logsumexp(logits, dim=-1, keepdim=True) # (B, seq_len, 1)

        return logits - logZ 
    
    def reset_parameters(self):
        self.token_embed.reset_parameters()
        self.time_embed.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.final_layer_norm.reset_parameters()
        if not self.tied_word_embeddings:
            self.output_proj.reset_parameters()

    def forward(
        self, 
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        start_pos: Optional[int] = None, 
        time: Optional[float] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass through the TokenFlow model. 
        Depending on the mode (training or inference), it processes the input tokens 
        or embeddings and computes the output logits and loss.
        Args:
            tokens_or_embeds (torch.Tensor): Input tokens or embeddings.
            labels (torch.Tensor, optional): Labels for training mode.
            start_pos (int, optional): Starting position for inference mode.
            time (float, optional): Flow time for inference.
        Returns:
            dict: Output logits and loss (if in training mode).
        """
        if labels is not None:
            return self.training_forward(input_ids, labels)

        return self.inference_forward(input_ids, start_pos, time)
        

    def training_forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass for training mode.
        Args:
            input_ids (torch.Tensor): Input data tokens.
            labels (torch.Tensor): Data labels for training.
        Returns:
            dict: Output logits and loss.
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len % self.blk_size == 0, f"Sequence length {seq_len} is not a multiple of block size {self.blk_size}."
        assert labels is not None, "Training mode requires labels."

        start_pos = 0
        freqs_cis = self._rope_cache(input_ids.device).repeat(2, 1)
        
        x1 = self.token_embed(input_ids)
        x0 = torch.randn_like(x1) * self.embed_scale
        # t_sample = self._sample_time((batch_size, self.blk_num), input_ids.device)
        t_sample = torch.rand((batch_size, self.blk_num), device=input_ids.device)
        t_all = torch.cat([torch.ones_like(t_sample), t_sample], dim=1)
        t_full = t_sample.repeat_interleave(self.blk_size, dim=1).unsqueeze(-1)
        
        h = rectified_flow_interpolate(x0, x1, t_full) # [x1, xt]
        t_vec = self.time_embed(timestep_embedding(t_all, self.time_dim))
        xt = h[:, -self.blk_num * self.blk_size:] # (batch_size, 2 * blk_num * blk_size, dim)

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=False)

        batch_size_, seqlen_ = labels.shape
        assert batch_size_ == batch_size and seqlen_ == seq_len, f"Labels must match the shape of input {input_ids.shape}. Got {labels.shape}."

        h = h[:, -self.blk_num * self.blk_size:]
        h = self.final_layer_norm(h)
        if self.tied_word_embeddings:
            model_logits = F.linear(h, self.token_embed.weight, None)
        else:
            model_logits = self.output_proj(h)
        singular_logits = self._compute_singular_logits(xt, t_full, std=self.embed_scale)
        logits = model_logits + singular_logits
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.reshape(-1), 
            ignore_index=-100, 
            reduction="mean"
        )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )
        

    @torch.inference_mode()
    def inference_forward(self, h: torch.Tensor, start_pos: int, time: float):
        """
        Forward pass for inference mode.
        Args:   
            h (torch.Tensor): Input tensor.
            start_pos (int): Starting position for inference.
            time (float): Flow time for inference.
        Returns:    
            dict: Output logits.
        """
        batch_size, seq_len = h.shape[0], h.shape[1]
        assert seq_len % self.blk_size == 0, f"Sequence length {seq_len} is not a multiple of block size {self.N}"
        assert start_pos is not None, "Inference mode requires start_pos."
        assert time is not None, "Inference mode requires time."

        freqs_cis = self._rope_cache(h.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seq_len]
        t_all = build_inference_time(time, seq_len, batch_size, self.blk_size).to(h.device)
        t_vec = self.time_embed(timestep_embedding(t_all, self.time_dim))
        t_full = torch.full((batch_size, self.blk_size), time, device=h.device).unsqueeze(-1)

        xt = h[:, -self.blk_size:]

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=True)
        
        h = h[:, -self.blk_size:]
        h = self.final_layer_norm(h)
        if self.tied_word_embeddings:
            model_logits = F.linear(h, self.token_embed.weight, None)
        else:
            model_logits = self.output_proj(h)
        
        singular_logits = self._compute_singular_logits(xt, t_full, std=self.embed_scale)
        logits = model_logits + singular_logits

        return {"logits": logits}
    
    
    @torch.inference_mode()
    def generate(
        self,
        batch_size: int,
        time_schedule: List[float],
        bos_id: int = 1,
        eos_id: int = 1,
    ) -> Tuple[List[List[int]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            time_schedule (List[float]): List of time steps for the generation process. Must start with 0 and end with 1.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 1.0 (switched off).
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
        Returns:
            Tuple[List[List[int]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        """
        assert batch_size <= self.max_batch, f"Generation batch size {batch_size} greater than batch size limit {self.max_batch}."
        assert all(0 <= x <= 1 for x in time_schedule), "Time steps must between 0 and 1."
        total_len = self.blk_num * self.blk_size

        device = self.token_embed.weight.device
        tokens = torch.full((batch_size, total_len), bos_id, dtype=torch.long, device=device)
        
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        prev_pos = 0
        for cur_pos in range(0, total_len, self.blk_size):
            x_all = torch.randn(batch_size, self.blk_size, self.dim, device=device) * self.embed_scale
            if cur_pos - prev_pos > 0:
                x1 = self.token_embed(tokens[:, prev_pos:cur_pos])
                x_all = torch.cat([x1, x_all], dim=1)

            for i, (time, next_time) in enumerate(zip(time_schedule[:-1], time_schedule[1:])):
                logits = self.forward(x_all, start_pos=prev_pos, time=time)["logits"]
                if i == 0:
                    prev_pos = cur_pos
                    x_all = x_all[:, -self.blk_size:]

                E = self.token_embed.weight
                probs = torch.softmax(logits[:, -self.blk_size:], dim=-1)
                x1t = torch.matmul(probs, E) # \Hat{X1} estimation at time t is exptectation of embedding vectors
                alpha = (next_time - time) / (1 - time) # (t_{i+1} - t_i) / (1 - t_i)
                x_all.lerp_(x1t, alpha)
                
            x1_flat = x_all.reshape(-1, self.dim)               # (B*N, d)

            dist = torch.cdist(x1_flat, E)         # default is p=2 (Euclidean)
            closest = dist.argmin(dim=1)           # (B*N,)
            next_tokens = closest.view(batch_size, self.blk_size)     
            tokens[:, cur_pos:cur_pos+self.blk_size] = next_tokens

            eos_in_block = (next_tokens == eos_id).any(dim=1)
            eos_reached |= eos_in_block

            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)

        return out_tokens