from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from model_utils import (
    init_normal,
    precompute_freqs_cis,
    apply_rotary_emb,
    timestep_embedding,
    rectified_flow_interpolate,
    sample_posterior_gumbel,
)


class TokenFlowConfig(PretrainedConfig):
    model_type = "tokenflow"  # used internally by HF for registry & save/load

    def __init__(
        self,
        ctx_len: int = 1024,
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
        # Attention parameters
        use_causal: bool = False,
        # Gumbel reflow parameters
        use_gumbel_flow: bool = False,
        gumbel_conditioning_type: str = "x0", # "x0" or "cross_attention"
        teacher_model_name: Optional[str] = None,
        freeze_token_embed: bool = False,  # Whether to freeze token embeddings after teacher initialization
        init_from_teacher: bool = False,    # Whether to initialize token embeddings from the teacher model
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ctx_len = ctx_len
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
        self.use_gumbel_flow = use_gumbel_flow
        self.gumbel_conditioning_type = gumbel_conditioning_type
        self.teacher_model_name = teacher_model_name
        self.use_causal = use_causal
        self.freeze_token_embed = freeze_token_embed
        self.init_from_teacher = init_from_teacher


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
            vec(torch.Tensor): (B, 1, dim) — one vector per block.
        Returns:
            ModulationOut: modulation parameters (shift, scale, gate) each of shape (B, 1, dim)
        """
        out = self.w(F.silu(vec))  # (B, 1, 3*dim)
        shift, scale, gate = out.chunk(3, dim=-1)  # each (B, 1, dim)
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
    Multi-head attention w/ rotary positional encodings + Flash Attention 2.
    Args:
        config (TokenFlowConfig): Model configuration parameters.
    Attributes:
        n_kv_heads (int): Number of key/value heads.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension size of each attention head.
        wq (nn.Linear): Linear transformation for query.
        wk (nn.Linear): Linear transformation for key.
        wv (nn.Linear): Linear transformation for value.
        wo (nn.Linear): Linear transformation for output.
    """

    def __init__(self, layer_id: int, config: TokenFlowConfig):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.max_batch = config.max_batch
        self.ctx_len = config.ctx_len
        self.init_cutoff_factor = config.init_cutoff_factor
        self.use_causal = config.use_causal

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.dim)
        attn_out_std = 1 / (math.sqrt(2 * self.dim * (self.layer_id + 1)))

        init_normal(self.wq, std, self.init_cutoff_factor)
        init_normal(self.wk, std, self.init_cutoff_factor)
        init_normal(self.wv, std, self.init_cutoff_factor)
        init_normal(self.wo, attn_out_std, self.init_cutoff_factor)

    def forward(self, x: Tensor, start_pos: int, freqs_cis, use_cache=False, kv_input: Optional[Tensor] = None):
        """
        Args
        ----
        x          : (B, L, D)
        start_pos  : starting token position in the *full* sequence
                     (needed only for KV-cache path)
        freqs_cis  : rotary embedding tensor
        use_cache  : whether to use KV caching for inference
        kv_input   : (B, L, D)
                     optional input for key and value for cross-attention
                     if None, performs self-attention
        """
        batch_size, seq_len, _ = x.shape

        q = self.wq(x)
        if kv_input is not None:
            # Cross-attention: k and v are from kv_input
            k = self.wk(kv_input)
            v = self.wv(kv_input)
        else:
            # Self-attention: k and v are from x
            k = self.wk(x)
            v = self.wv(x)

        # (B, L, H, D_h)
        q = q.view(batch_size, seq_len, self.n_heads,    self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if not use_cache:
            # Training path: use scaled dot-product attention
            # Transpose to (B, H, L, D_h) for scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=self.use_causal,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            # Transpose back to (B, L, H, D_h)
            out = out.transpose(1, 2)
        else:
            # Inference path with KV caching
            end_pos = start_pos + seq_len
            
            if self.cache_k.numel() == 0 or self.cache_k.size(1) < end_pos:
                max_len = self.ctx_len
                shape   = (self.max_batch, max_len, self.n_kv_heads, self.head_dim)
                self.cache_k = torch.zeros(shape, device=x.device, dtype=k.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                
            self.cache_k[:batch_size, start_pos:end_pos] = k
            self.cache_v[:batch_size, start_pos:end_pos] = v

            k_full = self.cache_k[:batch_size, :end_pos]
            v_full = self.cache_v[:batch_size, :end_pos]

            # Transpose to (B, H, L, D_h) for scaled_dot_product_attention
            q = q.transpose(1, 2)
            k_full = k_full.transpose(1, 2)
            v_full = v_full.transpose(1, 2)

            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                is_causal=self.use_causal,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            # Transpose back to (B, L, H, D_h)
            out = out.transpose(1, 2)

        # (B, L, D)
        out = out.reshape(batch_size, seq_len, -1)
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

    The normalization must be applied manually, for example, after each
    optimizer step.

    Attributes:
        weight (nn.Parameter): The learnable parameter tensor.
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
        Performs a forward pass using the stored weights.
        This method does NOT perform weight normalization.
        """
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
            self_attention (Attention): Self-attention module.
            cross_attention (Attention): Cross-attention module for conditioning.
            feed_forward (FeedForward): FeedForward module.
            attention_norm (RMSNorm): Layer normalization for self-attention output.
            cross_attention_norm (RMSNorm): Layer normalization for cross-attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            modulation (Modulation): adaLN modulation module.
        """
        super().__init__()
        self.self_attention = Attention(layer_id, config)
        self.cross_attention = Attention(layer_id, config)
        self.feed_forward = FeedForward(layer_id, config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.cross_attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.modulation = Modulation(config)

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.cross_attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.cross_attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.modulation.reset_parameters()
    
    def forward(
        self, 
        x: torch.Tensor, 
        vec: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        use_cache: bool = False,
        gumbel_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mod = self.modulation(vec) # (B, 1, dim)

        # 1. Self-Attention
        x_mod_self = (1 + mod.scale) * self.attention_norm(x) + mod.shift
        h = x + self.self_attention(x_mod_self, start_pos, freqs_cis, use_cache)

        # 2. Cross-Attention (only if gumbel_context is provided)
        if gumbel_context is not None:
            x_mod_cross = (1 + mod.scale) * self.cross_attention_norm(h) + mod.shift
            h = h + self.cross_attention(x_mod_cross, start_pos, freqs_cis, use_cache, kv_input=gumbel_context)

        # 3. Feed-Forward
        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + mod.gate * ffn_out # broadcast gate over seq len

        return out


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
        self.ctx_len = config.ctx_len
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.max_batch = config.max_batch
        self.dim = config.dim
        self.time_dim = config.time_dim
        self.embed_scale = config.embed_scale
        self.tied_word_embeddings = config.tie_word_embeddings

        self.token_embed = NormalizedEmbedding(config)
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
            self.dim//self.n_heads, self.ctx_len, config.rope_scaling)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Teacher model for Gumbel reflow
        if config.use_gumbel_flow:
            # Only load teacher model if teacher_model_name is provided
            # During inference, we may not need the teacher model
            if config.teacher_model_name:
                print(f"Loading teacher model: {config.teacher_model_name}")
                from transformers import AutoModelForCausalLM
                # load local or hub checkpoints
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    config.teacher_model_name,
                )
                teacher_model.requires_grad_(False)
                teacher_model.eval()
                
                # Store teacher model in a way that won't be registered as a submodule
                # Use object.__setattr__ to avoid PyTorch's module registration
                object.__setattr__(self, '_teacher_model', teacher_model)
                
                # Count teacher model parameters
                teacher_params = sum(p.numel() for p in teacher_model.parameters())
                print(f"Teacher model loaded successfully! ({teacher_params:,} parameters)")
                print("Teacher model set to eval mode with requires_grad=False")

                if config.init_from_teacher:
                    # Initialize token_embed from teacher model embeddings
                    print("Initializing token_embed from teacher model embeddings...")
                    teacher_embeddings = teacher_model.get_input_embeddings().weight
                    
                    # Copy teacher embeddings to token_embed
                    with torch.no_grad():
                        self.token_embed.weight.data.copy_(teacher_embeddings)
                        
                    print("Copied teacher embeddings (preserving original scale)")
                    print(f"token_embed.weight.requires_grad: {self.token_embed.weight.requires_grad}")
                    
                    # Optionally freeze token embeddings
                    if config.freeze_token_embed:
                        self.token_embed.weight.requires_grad_(False)
                        print("Frozen token_embed weights (requires_grad=False)")
                        print("Token embeddings will NOT be updated during training")
                    else:
                        print("Token embeddings remain trainable (will be fine-tuned during training)")
                else:
                    print("Skipping teacher embedding initialization (using random initialization).")
            
            else:
                print("Note: Gumbel flow enabled but no teacher model specified (inference mode)")

    
    @property
    def teacher_model(self):
        """Access teacher model without it being registered as a submodule."""
        return getattr(self, '_teacher_model', None)
    
    def _ensure_teacher_device(self, device):
        """Ensure teacher model is on the same device as the main model."""
        if hasattr(self, '_teacher_model') and self._teacher_model is not None:
            if next(self._teacher_model.parameters()).device != device:
                self._teacher_model = self._teacher_model.to(device)
    
    def _rope_cache(self, device):
        if (not hasattr(self, "_freqs_cis") or self._freqs_cis.device != device):
            freqs_cis = precompute_freqs_cis(
                self.dim // self.n_heads,
                self.ctx_len,
                self.config.rope_scaling,
            ).to(device)
            self.register_buffer("_freqs_cis", freqs_cis, persistent=False)
        return self._freqs_cis
        
        
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
        assert labels is not None, "Training mode requires labels."

        start_pos = 0
        freqs_cis = self._rope_cache(input_ids.device)
        
        x1 = self.token_embed(input_ids)
        
        # Posterior Gumbel reflow path
        if self.config.use_gumbel_flow:
            # gumbel_type == "expected_embeddings" (current)
            # Could try different x0 and x1 here, 
            # Using x0 = softmax(gumbel) @ embed_weight and x1 = embed(input_ids) for now
            # Alternatives: (need input projection, and changes in the generate function)
            # gumbel_type == "softmax_gumbel", x0 = softmax(gumbel), x1 = one-hot(input_ids)
            # gumbel_type == "gumbel_log_onehot", x0 = gumbel, x1 = log(one-hot(input_ids)) 
            assert self.teacher_model is not None, "Teacher model should be loaded when use_gumbel_flow=True"
            
            # Ensure teacher model is on the same device as input_ids
            self._ensure_teacher_device(input_ids.device)
            
            with torch.no_grad():
                teacher_output = self.teacher_model(input_ids)
                logits = teacher_output.logits.to(torch.float32)

            gumbel_noise = sample_posterior_gumbel(logits, input_ids)  # (B, T, V)
            gumbel_probs = F.softmax(gumbel_noise, dim=-1)  # (B, T, V)
            # Use einsum for efficient matrix multiplication: (B, T, V) @ (V, D) -> (B, T, D)
            gumbel_embeddings = torch.einsum('btv,vd->btd', gumbel_probs, self.token_embed.weight)

            gumbel_context = None
            if self.config.gumbel_conditioning_type == 'cross_attention':
                # Use Gaussian noise for x0
                x0 = torch.randn_like(x1) * self.embed_scale
                # Use Gumbel embeddings as context, detached from the graph
                gumbel_context = gumbel_embeddings.detach()
            else: # 'x0' or default
                # Use Gumbel embeddings for x0
                x0 = gumbel_embeddings
        
        # Original path
        else:
            x0 = torch.randn_like(x1) * self.embed_scale
            
        t_sample = torch.rand((batch_size, 1), device=input_ids.device)
        t_full = t_sample.repeat(1, seq_len).unsqueeze(-1)
        
        h = rectified_flow_interpolate(x0, x1, t_full)
        t_vec = self.time_embed(timestep_embedding(t_sample, self.time_dim))
        xt = h

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=False, gumbel_context=gumbel_context)

        batch_size_, seq_len_ = labels.shape
        assert batch_size_ == batch_size and seq_len_ == seq_len, f"Labels must match the shape of input {input_ids.shape}. Got {labels.shape}."

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
    def inference_forward(self, h: torch.Tensor, start_pos: int, time: float, gumbel_context: Optional[torch.Tensor] = None):
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
        assert start_pos is not None, "Inference mode requires start_pos."
        assert time is not None, "Inference mode requires time."

        freqs_cis = self._rope_cache(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

        t_sample = torch.full((batch_size, 1), time, device=h.device)
        t_vec = self.time_embed(timestep_embedding(t_sample, self.time_dim))
        xt = h

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, use_cache=True, gumbel_context=gumbel_context)

        h = self.final_layer_norm(h)
        if self.tied_word_embeddings:
            model_logits = F.linear(h, self.token_embed.weight, None)
        else:
            model_logits = self.output_proj(h)

        # Expand t_sample to match sequence length for _compute_singular_logits
        t_full = t_sample.repeat(1, seq_len).unsqueeze(-1)  # (B, seq_len, 1)
        singular_logits = self._compute_singular_logits(xt, t_full, std=self.embed_scale)
        logits = model_logits + singular_logits

        return CausalLMOutput(logits=logits)
    
    
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
        
        total_len = self.ctx_len
        device = self.token_embed.weight.device

        gumbel_context = None
        # Initialize x_t based on whether we're using Gumbel reflow or not
        if self.config.use_gumbel_flow:
            if self.config.gumbel_conditioning_type == 'cross_attention':
                # Sample from the Gumbel prior to create the context
                gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, total_len, self.config.vocab_size, device=device) + 1e-20) + 1e-20)
                gumbel_probs = F.softmax(gumbel_noise, dim=-1)
                gumbel_context = torch.einsum('btv,vd->btd', gumbel_probs, self.token_embed.weight).detach()

                # Initialize the actual state x_t with Gaussian noise
                x_t = torch.randn(batch_size, total_len, self.dim, device=device) * self.embed_scale
            else: # 'x0' or default
                # Original behavior: sample from Gumbel prior and use for x_t
                gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, total_len, self.config.vocab_size, device=device) + 1e-20) + 1e-20)
                gumbel_probs = F.softmax(gumbel_noise, dim=-1)
                x_t = torch.einsum('btv,vd->btd', gumbel_probs, self.token_embed.weight)
        else:
            # Original random noise path
            x_t = torch.randn(batch_size, total_len, self.dim, device=device) * self.embed_scale

        for i, (time, next_time) in enumerate(zip(time_schedule[:-1], time_schedule[1:])):
            logits = self.inference_forward(x_t, start_pos=0, time=time, gumbel_context=gumbel_context).logits
            E = self.token_embed.weight
            probs = torch.softmax(logits, dim=-1)
            x1t = torch.matmul(probs, E)
            alpha = (next_time - time) / (1 - time)
            x_t.lerp_(x1t, alpha)
        
        x1_flat = x_t.reshape(-1, self.dim)

        dist = torch.cdist(x1_flat, self.token_embed.weight)
        closest = dist.argmin(dim=1)
        tokens = closest.view(batch_size, total_len)

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)

        return out_tokens