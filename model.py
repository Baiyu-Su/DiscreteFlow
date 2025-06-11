from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import OrderedDict
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
    row_rms
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
        time_dim: int = 256,
        n_heads: int = 16,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 12,
        init_cutoff_factor: float = 3.0,
        tie_word_embeddings = True,
        rope_scaling: int = 10000,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        load_stats: bool = True,
        **kwargs,                      # catch any additional HF args
    ):
        super().__init__(**kwargs)

        self.blk_num = blk_num
        self.blk_size = blk_size
        self.max_batch = max_batch
        self.vocab_size = vocab_size
        self.dim = dim
        self.time_dim = time_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_layers = n_layers
        self.init_cutoff_factor = init_cutoff_factor
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
    def __init__(self, dim: int):
        """
        Initialize the adaLN Modulation layer.
        Args:
            dim (int): Model dimension.
        Attributes:
            w (nn.Linear): Linear transformation for modulation (3 terms).
        """
        super().__init__()
        self.w = nn.Linear(dim, 3 * dim)

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
    def __init__(self, in_dim: int, dim: int):
        """
        Initialize the MLPEmbedder module.
        Args:
            in_dim (int): Input dimension of time vector.
            dim (int): Model dimension.
        Attributes:
            in_layer (nn.Linear): Linear transformation for input.
            silu (nn.SiLU): Swish activation function.
            out_layer (nn.Linear): Linear transformation for output.
        """
        super().__init__()
        self.wi = nn.Linear(in_dim, dim, bias=True)
        self.silu = nn.SiLU()
        self.wo = nn.Linear(dim, dim, bias=True)

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

        # Projections ---------------------------------------------------------
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
    def __init__(
        self,
        layer_id: int,
        config: TokenFlowConfig,
    ):
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
    """Embedding layer whose vectors always have unit RMS.

    * ``weight_param`` is the single learnable **nn.Parameter**.
    * ``weight`` is a *property* that returns the RMS‑normalised tensor.
    * ``state_dict`` is patched so the saved tensor is already normalised.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 *,
                 padding_idx: int | None = None,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 eps: float = 1e-12):
        super().__init__()

        self.raw_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.register_parameter('weight_param', self.raw_weight)  # explicit alias

        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.eps = eps

        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:  # noqa: D401
        """RMS-normalised tensor view."""
        return self.raw_weight / row_rms(self.raw_weight, self.eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.embedding(
            input,
            self.weight,                       # unit‑RMS
            padding_idx=self.padding_idx,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def state_dict(self, destination: OrderedDict[str, torch.Tensor] | None = None,
                   prefix: str = '',
                   keep_vars: bool = False) -> OrderedDict[str, torch.Tensor]:  # type: ignore[override]
        sd = super().state_dict(destination, prefix, keep_vars=True)
        key = prefix + 'raw_weight'
        if key in sd:
            w = self.weight  # detached normalised tensor
            sd[key] = w if keep_vars else w.detach().clone()
        return sd if keep_vars else OrderedDict((k, v.detach().clone()) for k, v in sd.items())

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):  # noqa: D401,E501
        key = prefix + 'raw_weight'
        if key in state_dict:
            t = state_dict[key]
            if abs((t.pow(2).mean(1).sqrt()).mean().item() - 1.0) < 1e-3:
                state_dict[key] = t  # already normalised
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self) -> None:  # noqa: D401
        nn.init.normal_(self.raw_weight, mean=0.0, std=1.0 / math.sqrt(self.raw_weight.size(1)))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.raw_weight[self.padding_idx].fill_(0)


class SharedNormalizedLinear(nn.Module):
    def __init__(self, embed: NormalizedEmbedding, *, bias: bool = False):
        super().__init__()
        self.embed = embed
        if bias:
            self.bias = nn.Parameter(torch.zeros(embed.weight_param.size(0)))
        else:
            self.register_parameter('bias', None)

    @property
    def weight(self):  # always RMS‑normalised
        return self.embed.weight

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    

class TokenFlowBlock(nn.Module):
    def __init__(self, layer_id: int, config: TokenFlowConfig):
        """
        Initialize a TokenFlowBlock.
        Args:
            layer_id (int): Identifier for the layer.
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            N (int): Block size.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            modulation (Modulation): adaLN modulation module.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.blk_size = config.blk_size
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4*config.dim,
            multiple_of=config.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.modulation = Modulation(config.dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        vec: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        use_cache: bool = False
    ) -> torch.Tensor:
        mod = self.modulation(vec)                        # (B, 2*M, dim)

        B, ctx, D = x.shape
        x_blk      = x.view(B, -1, self.blk_size, D)       # (B, 2*M, N, D)
        mod_shift  = mod.shift.unsqueeze(2)                 # (B, 2*M, 1, D)
        mod_scale  = mod.scale.unsqueeze(2)
        mod_gate   = mod.gate .unsqueeze(2)                 # idem

        x_mod = (1 + mod_scale) * self.attention_norm(x_blk) + mod_shift
        x_mod_flat = x_mod.view(B, ctx, D)                  # (B, ctx, D)
        h = x + self.attention(x_mod_flat, start_pos, freqs_cis, use_cache)

        ffn_out_blk = self.feed_forward(self.ffn_norm(h)).view(
            B, -1, self.blk_size, D
        )                                                   # (B, 2*M, N, D)

        h_blk  = h.view(B, -1, self.blk_size, D)
        out_blk = h_blk + mod_gate * ffn_out_blk            # broadcast over N

        return out_blk.view(B, ctx, D)


class TokenFlowModel(PreTrainedModel):
    def __init__(self, config: TokenFlowConfig):
        """
        Initialize a Token Flow Model.
        Args:
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
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
        self.tied_word_embeddings = config.tie_word_embeddings

        self.token_embed = NormalizedEmbedding(config.vocab_size, config.dim)
        # self.beta_dist = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(5.0))
        self.time_embed = MLPEmbedder(in_dim=self.time_dim, dim=config.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TokenFlowBlock(layer_id, config))

        if config.tie_word_embeddings:
            self.final_layer_norm = nn.LayerNorm(
                config.dim,
                eps=config.norm_eps,
                elementwise_affine=True          # keep it learnable
            )
            # with torch.no_grad():
            #     self.final_layer_norm.weight.fill_(0.5)
            self.output_proj = SharedNormalizedLinear(self.token_embed, bias=False)
            self._tied_weights_keys = [r"^token_embed\.weight_param$", r"^output_proj\.embed\.weight_param$"]
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
    
        
    def _compute_singular_logits(self, xt: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        xt:     (B, seq_len=M*N, d)
        t_full: (B, seq_len)           -- per-position time
        returns flow_logits of shape (B, seq_len, V)
        """
        E = self.token_embed.weight
        vocab_size, dim = E.shape

        xt_norm2 = xt.pow(2).sum(dim=-1, keepdim=True) # (B, seq_len, 1)
        e_norm2 = E.pow(2).sum(dim=-1).unsqueeze(0).unsqueeze(0)

        cross = xt @ E.t() # (B, seq_len, V)
        D = xt_norm2 - 2 * t * cross + (t**2) * e_norm2  # (B, seq_len, V)
        denom = 2 * (1 - t).pow(2) + eps ** 2 # (B, seq_len, 1)
        D_scaled = -D / denom # (B, seq_len, V)

        logZ = torch.logsumexp(D_scaled, dim=-1, keepdim=True) # (B, seq_len, 1)

        return D_scaled - logZ
    
    
    # def _tied_weights_keys(self):
    #     """
    #     Return a list whose elements are *sets* of parameter names that
    #     share the very same storage.  The saver compares its findings
    #     against this list and raises if they diverge.
    #     """
    #     return [{"token_embed.weight", "output_proj.weight"}]
    

    def forward(
        self, 
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        start_pos: Optional[int] = None, 
        time: Optional[float] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass through the TokenFlow model. Depending on the mode (training or inference), it processes the input tokens or embeddings and computes the output logits and loss.
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
        x0 = torch.randn_like(x1)
        # t_sample = self._sample_time((batch_size, self.blk_num), input_ids.device)
        t_sample = torch.rand((batch_size, self.blk_num), device=input_ids.device)
        t_all = torch.cat([torch.ones_like(t_sample), t_sample], dim=1)
        t_full = t_sample.repeat_interleave(self.blk_size, dim=1).unsqueeze(-1)
        
        x_all = rectified_flow_interpolate(x0, x1, t_full) # [x1, xt]
        t_vec = self.time_embed(timestep_embedding(t_all, self.time_dim))
        xt = x_all[:, -self.blk_num * self.blk_size:] # (batch_size, 2 * blk_num * blk_size, dim)
        h = x_all

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=False)

        batch_size_, seqlen_ = labels.shape
        assert batch_size_ == batch_size and seqlen_ == seq_len, f"Labels must match the shape of input {input_ids.shape}. Got {labels.shape}."

        h = h[:, -self.blk_num * self.blk_size:]
        h = self.final_layer_norm(h)
        model_logits = self.output_proj(h)
        
        singular_logits = self._compute_singular_logits(xt, t_full)
        logits = model_logits + singular_logits
        
        # if self.load_stats:           
        #     with torch.no_grad():
        #         K = 10
        #         mdl_flat = model_logits.detach().abs().view(-1, model_logits.size(-1))
        #         sng_flat = singular_logits.detach().abs().view(-1, singular_logits.size(-1))
        #         all_flat = logits.detach().abs().view(-1, singular_logits.size(-1))

        #         mdl_topk_mean = mdl_flat.topk(K, dim=-1).values.mean().item()
        #         sng_topk_mean = sng_flat.topk(K, dim=-1).values.mean().item()
        #         all_topk_mean = all_flat.topk(K, dim=-1).values.mean().item()

        #         self._logits_stats = {
        #             "logits/model_top10_mean":     mdl_topk_mean,
        #             "logits/model_max":           mdl_flat.max().item(),
        #             "logits/singular_top10_mean":  sng_topk_mean,
        #             "logits/singular_max":        sng_flat.max().item(),
        #             "logits/all_top10_mean":  all_topk_mean,
        #             "logits/all_max":        all_flat.max().item(),
        #         }
        
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
    def inference_forward(self, x_all: torch.Tensor, start_pos: int, time: float):
        """
        Forward pass for inference mode.
        Args:   
            h (torch.Tensor): Input tensor.
            start_pos (int): Starting position for inference.
            time (float): Flow time for inference.
        Returns:    
            dict: Output logits.
        """
        batch_size, seq_len = x_all.shape[0], x_all.shape[1]
        assert seq_len % self.blk_size == 0, f"Sequence length {seq_len} is not a multiple of block size {self.N}"
        assert start_pos is not None, "Inference mode requires start_pos."
        assert time is not None, "Inference mode requires time."

        freqs_cis = self._rope_cache(x_all.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seq_len]
        t_all = build_inference_time(time, seq_len, batch_size, self.blk_size).to(x_all.device)
        t_vec = self.time_embed(timestep_embedding(t_all, self.time_dim))
        t_full = torch.full((batch_size, self.blk_size), time, device=x_all.device).unsqueeze(-1)

        xt = x_all[:, -self.blk_size:]
        h = x_all

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=True)
        
        h = h[:, -self.blk_size:]
        h = self.final_layer_norm(h)
        model_logits = self.output_proj(h)
        
        singular_logits = self._compute_singular_logits(xt, t_full)
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
            x_all = torch.randn(batch_size, self.blk_size, self.dim, device=device)
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