from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from typing import Tuple, Optional, Union

from utils import (
    precompute_freqs_cis,
    apply_rotary_emb,
    build_training_mask,
    build_inference_mask,
    timestep_embedding,
    repeat_kv,
    nucleus_cutoff,
    build_time_tensor,
)


class TokenFlowConfig(PretrainedConfig):
    model_type = "tokenflow"  # used internally by HF for registry & save/load

    def __init__(
        self,
        is_inference: bool = False,
        M: int = 128,
        N: int = 8,
        max_B: int = 32,
        vocab_size: int = 32001,
        dim: int = 1024,
        time_dim: int = 256,
        n_heads: int = 16,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 12,
        rope_scaling: int = 10000,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        **kwargs,                      # catch any additional HF args
    ):
        # this handles e.g. `id2label`, `label2id`, `torch_dtype`, etc.
        super().__init__(**kwargs)

        # now assign your own fields
        self.is_inference = is_inference
        self.M            = M
        self.N            = N
        self.max_B        = max_B
        self.vocab_size   = vocab_size
        self.dim          = dim
        self.time_dim     = time_dim
        self.n_heads      = n_heads
        self.n_kv_heads   = n_kv_heads or n_heads
        self.n_layers     = n_layers
        self.rope_scaling = rope_scaling
        self.multiple_of  = multiple_of
        self.norm_eps     = norm_eps


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
    """Multi-head attention module."""

    def __init__(self, config: TokenFlowConfig):
        """
        Initialize the Attention module.
        Args:
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads (int): Number of attention heads.
            n_rep (int): Number of repetitions for kv heads.
            head_dim (int): Dimension size of each attention head.
            N (int): Block size.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
            is_inference (bool): Flag for inference mode.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
        """
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_heads = config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.N = config.N

        self.wq = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.n_heads * self.head_dim,
            config.dim,
            bias=False,
        )

        self.is_inference = config.is_inference
        if self.is_inference:
            self.cache_k = torch.zeros(
                (
                    config.max_B,
                    config.M*config.N,
                    self.n_kv_heads,
                    self.head_dim,
                ),
                device="cuda",
            )
            self.cache_v = torch.zeros(
                (
                    config.max_B,
                    config.M*config.N,
                    self.n_kv_heads,
                    self.head_dim,
                ),
                device="cuda",
            )
        else:
            self.cache_k = None
            self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        B, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape into heads
        xq = xq.view(B, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        del start_pos
        keys, values = xk, xv

        # expand kv heads if needed
        keys   = repeat_kv(keys,   self.n_rep)  # → (B, Lk, H, D)
        values = repeat_kv(values, self.n_rep)  # → (B, Lk, H, D)

        # transpose into (B, H, L, D)
        q = xq.transpose(1, 2)     # (B, H, L, D)
        k = keys.transpose(1, 2)   # (B, H, Lk, D)
        v = values.transpose(1, 2) # (B, H, Lk, D)

        # free big tensors asap
        del xq, keys, values

        output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=0.0,
        )

        # back to (B, L, H, D) → combine heads → project
        output = output.transpose(1, 2).reshape(B, seq_len, -1)
        return self.wo(output)



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        """
        Initialize the FeedForward module.
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        Attributes:
            w1 (nn.Linear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third layer.
        """
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        """
        Forward pass of the FeedForward module.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying feedforward layers.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
        self.N = config.N
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=int(3.5*config.dim),
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
    ):
        """
        Perform a forward pass through the TokenFlowBlock.
        Args:
            x (torch.Tensor): Input tensor.
            vec (torch.Tensor): Time modulation tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        mod = self.modulation(vec)  # mod.shift, mod.scale, mod.gate (B, 2*M, dim)

        mod_shift = mod.shift.repeat_interleave(self.N, dim=1)  # (B, ctx, dim)
        mod_scale = mod.scale.repeat_interleave(self.N, dim=1)  # (B, ctx, dim)
        mod_gate = mod.gate.repeat_interleave(self.N, dim=1)   # (B, ctx, dim)

        x_mod = (1 + mod_scale) * self.attention_norm(x) + mod_shift
        h = x + self.attention(x_mod, start_pos, freqs_cis)
        out = h + mod_gate * self.feed_forward(self.ffn_norm(h))

        return out


class TokenFlowModel(PreTrainedModel):
    def __init__(self, config: TokenFlowConfig):
        """
        Initialize a Token Flow Model.
        Args:
            config (TokenFlowConfig): Model configuration parameters.
        Attributes:
            n_layers (int): Number of layers in the model.
            n_heads (int): Number of attention heads.
            M (int): Number of blocks.
            N (int): Block size.
            max_B (int): Maximum batch size for inference.
            dim (int): Model dimension.
            time_dim (int): Dimension of the time vector.
            is_inference (bool): Flag for inference mode.
            token_embed (nn.Embedding): Token embedding layer.
            time_embed (MLPEmbedder): Time embedding layer.
            mask (torch.Tensor): Masking tensor for attention.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            final_layer_norm (RMSNorm): Final layer normalization.
            output_proj (nn.Linear): Output projection layer.
            freqs_cis (torch.Tensor): Precomputed frequency tensors.
        """
        super().__init__(config)
        self.config = config
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.M = config.M
        self.N = config.N
        self.max_B = config.max_B
        self.dim = config.dim
        self.time_dim = config.time_dim

        self.is_inference = config.is_inference
        self.token_embed = nn.Embedding(config.vocab_size, self.dim)
        self.time_embed = MLPEmbedder(in_dim=self.time_dim, dim=config.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TokenFlowBlock(layer_id, config))

        self.final_layer_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output_proj = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(
            self.dim//self.n_heads, self.M*self.N, config.rope_scaling)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.beta_dist = None
    
    def embed_for_training(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds clean input tokens and appends random noise embeddings.
        Args:
            input_ids (torch.Tensor): Tensor of shape (B, seq_len) containing token indices.
            token_embed (nn.Module): Embedding module (nn.Embedding) that converts token indices to embeddings.
            time_embed (nn.Module): Embedding module for time embeddings.
            M (int): Number of blocks in the sequence.
            N (int): Generation block size.
            time_dim (int): Dimension of the time embedding.
        Returns:    
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - X_all (torch.Tensor): Concatenated tensor of shape (B, 2*M*N, dim) where dim is model dimension.
                - t_vec (torch.Tensor): Time embeddings of shape (B, 2*M, time_dim).
        """
        X1 = self.token_embed(input_ids)  # (B, M*N, dim)
        B = X1.shape[0]

        # sample (B, M) from Beta(2,5)
        t_sample = torch.rand((B, self.M)).to(X1.device)  # (B, M)
        t_full = t_sample.repeat_interleave(self.N, dim=1).unsqueeze(-1) # (B, M*N, 1)
        X0 = 0.077303 * torch.randn_like(X1) # (B, M*N, dim)
        Xt = t_full * X1 + (1 - t_full) * X0 # (B, M*N, dim)

        t_vec = self.time_embed(timestep_embedding(t_sample, self.time_dim))  # (B, 2*M, dim)

        return Xt, t_vec, t_full
    

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
        if self.is_inference:
            return self.inference_forward(input_ids, start_pos, time)
        else:
            return self.training_forward(input_ids, labels)

    def training_forward(self, tokens: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass for training mode.
        Args:
            tokens (torch.Tensor): Input data tokens.
            labels (torch.Tensor): Data labels for training.
        Returns:
            dict: Output logits and loss.
        """
        B, seq_len = tokens.shape
        assert seq_len % self.N == 0, f"Sequence length {seq_len} is not a multiple of block size {self.N}"
        assert labels is not None, "Training mode requires labels."

        start_pos = 0
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        h, t_vec, t_full = self.embed_for_training(tokens)
        # xt = h.detach().clone()

        # freqs_cis = self.freqs_cis.repeat(2, 1)

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, self.freqs_cis)

        B_, seqlen_ = labels.shape
        assert B_ == B and seqlen_ == seq_len, f"Labels must match the shape of input {tokens.shape}. Got {labels.shape}"

        h = self.final_layer_norm(h)
        logits = self.output_proj(h)
        # with torch.no_grad():
        #     flowlogits = self.compute_flow_logits(xt, t_full)
            
        # logits = logits - flowlogits
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.reshape(-1), 
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
        B, seq_len, _ = h.shape
        assert seq_len % self.N == 0, f"Sequence length {seq_len} is not a multiple of block size {self.N}"
        assert start_pos is not None, "Inference mode requires start_pos."
        assert time is not None, "Inference mode requires time."

        self.freqs_cis = self.freqs_cis.to(h.device)
        time_tensor = build_time_tensor(time, seq_len, B, self.N).to(h.device)
        t_vec = self.time_embed(timestep_embedding(time_tensor, self.time_dim))

        freqs_cis = self.freqs_cis[:seq_len]

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis)

        h = self.final_layer_norm(h)
        logits = self.output_proj(h)
            
        return {"logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        time_schedule: List[float],
    ) -> Tuple[List[List[int]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            time_schedule (List[float]): List of time steps for the generation process. Must start with 0 and end with 1.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 1.0 (switched off).
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
        Returns:
            Tuple[List[List[int]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        """
        assert all(0 <= x <= 1 for x in time_schedule), "Time steps must between 0 and 1."

        # 6) Build your tensor of shape (B, trunc_len) — no padding needed
        device = "cuda"  # or wherever your model lives
            
        E = self.token_embed.weight
        Xt = 0.077303 * torch.randn(4, self.N, self.dim, device=device)
        for i, (time, next_time) in enumerate(zip(time_schedule[:-1], time_schedule[1:])):
            logits = self.forward(Xt, start_pos=0, time=time)["logits"]

            probs = torch.softmax(logits, dim=-1)
            print(f"Shape of probs {probs.shape}, shape of E {E.shape}")
            
            X1t = torch.matmul(probs, E) # \Hat{X1} estimation at time t is exptectation of embedding vectors
            alpha = (next_time - time) / (1 - time) # (t_{i+1} - t_i) / (1 - t_i)
            print(alpha)
            Xt = alpha * X1t + (1 - alpha) * Xt  # \Hat{X1} at t_{i+1} is linear interpolation between \Hat{X1} at t_i and X0
       
        final_logits = self.forward(Xt, start_pos=0, time=1.0)["logits"]
        # torch.set_printoptions(threshold=float('inf'))
        # print(final_logits)
        
        next_tokens = final_logits.argmax(dim=-1)
        print(next_tokens.shape)
        return next_tokens.tolist()
