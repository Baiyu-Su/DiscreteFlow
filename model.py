from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        multiple_of: int = 256,
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
        mask: torch.Tensor,
    ):
        B, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape into heads
        xq = xq.view(B, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.is_inference:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            block_len = seq_len - self.N
            if block_len > 0:
                self.cache_k[:B, start_pos:start_pos+block_len] = xk[:, :block_len]
                self.cache_v[:B, start_pos:start_pos+block_len] = xv[:, :block_len]

            keys   = self.cache_k[:B, :start_pos+seq_len]
            values = self.cache_v[:B, :start_pos+seq_len]
        else:
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

        # now call the built‑in SDPA
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,    # broadcasts to (B, H, L, Lk)
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
        hidden_dim = int(2 * hidden_dim / 3)

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
        mask: torch.Tensor,
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
        h = x + self.attention(x_mod, start_pos, freqs_cis, mask)
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

        if not self.is_inference:
            mask_2d = build_training_mask(config.M, config.N).to(torch.bfloat16)  # shape (2MN, 2MN)
            mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 2MN, 2MN)
            self.register_buffer("mask", mask_4d, persistent=False)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TokenFlowBlock(layer_id, config))

        self.final_layer_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output_proj = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(
            self.dim//self.n_heads, self.M*self.N, config.rope_scaling)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    
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
        t_sample = torch.empty((B, self.M),
                               device=X1.device,
                               dtype=X1.dtype).beta_(2.0, 5.0)  # (B, M)
        t_full = t_sample.repeat_interleave(self.N, dim=1).unsqueeze(-1) # (B, M*N, 1)
        X0 = 0.0367 * torch.randn_like(X1) # (B, M*N, dim)
        Xt = t_full * X1 + (1 - t_full) * X0 # (B, M, N, dim)

        X_all = torch.cat([X1, Xt], dim=1)  # (B, 2*M*N, dim)
        t1 = torch.ones_like(t_sample)  # (B, M)
        t_all = torch.cat([t1, t_sample], dim=1)  # (B, 2*M)
        t_vec = self.time_embed(timestep_embedding(t_all, self.time_dim))  # (B, 2*M, dim)

        return X_all, t_vec, t_full

    
    def embed_for_inference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embeds clean input tokens and appends random noise embeddings.
        Args:
            input_ids (torch.Tensor): Tensor of shape (B, seq_len) containing token indices.
            token_embed (nn.Module): Embedding module (e.g., nn.Embedding) that converts token indices to embeddings.
            N (int): Generation block size.
        Returns:
            torch.Tensor: Concatenated tensor of shape (B, seq_len+N, dim) where dim is the embedding dimension.
        """
        embedded_tokens = self.token_embed(input_ids) #(B, seqlen, dim)
        B, _, dim = embedded_tokens.shape

        X0 = 0.0367 * torch.randn(B, self.N, dim, device=input_ids.device)
        combined_embeddings = torch.cat((embedded_tokens, X0), dim=1) # (B, seqlen+N, dim)

        return combined_embeddings
    
        
    def compute_flow_logits(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        xt:     (B, seq_len=M*N, d)
        t_full: (B, seq_len)           -- per‐position time
        returns flow_logits of shape (B, seq_len, V)
        """
        E = self.token_embed.weight # (V, d)
        V, d = E.shape

        xt_norm2 = xt.pow(2).sum(dim=-1, keepdim=True) # (B, seq_len, 1)
        E_norm2 = E.pow(2).sum(dim=-1).view(1, 1, V) # (1, 1, V)

        cross = xt @ E.t() # (B, seq_len, V)
        # print(f"shape of t: {t.shape}, shape of cross: {cross.shape}, shape of xt_norm2: {xt_norm2.shape}, shape of E_norm2: {E_norm2.shape}")
        D = xt_norm2 - 2 * t * cross + (t**2) * E_norm2  # (B, seq_len, V)
        denom = 2 * (1 - t).pow(2) # (B, seq_len, 1)
        D_scaled = D / denom # (B, seq_len, V)

        logZ = torch.logsumexp(-D_scaled, dim=-1, keepdim=True) # (B, seq_len, 1)

        return D_scaled + logZ
        

    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
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
            del attention_mask
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
        xt = h[:, self.M * self.N:, :].detach().clone()

        freqs_cis = self.freqs_cis.repeat(2, 1)
        mask = self.mask

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, mask)

        B_, seqlen_ = labels.shape
        assert B_ == B and seqlen_ == seq_len, f"Labels must match the shape of input {tokens.shape}. Got {labels.shape}"

        h = h[:, self.M * self.N:, :]
        h = self.final_layer_norm(h)
        logits = self.output_proj(h)
        with torch.no_grad():
            flowlogits = self.compute_flow_logits(xt, t_full)
            
        logits.sub_(flowlogits)

        # ce_tokenwise = F.cross_entropy(
        #     logits.view(-1, logits.size(-1)),       # (B*M*N, vocab_size)
        #     labels.reshape(-1),                # (B*M*N,)
        #     ignore_index=-100,
        #     reduction='none'
        # )
        # del logits
        # ce_tokenwise = ce_tokenwise.view(B, self.M * self.N)  # shape (B, M*N)

        # weights = (1.0 / t_full.clamp_min(1e-8).squeeze()).clamp_max(100.0)
        # weighted_ce = ce_tokenwise * weights
        # valid_mask = (labels != -100).float()
        # weighted_loss_sum = (weighted_ce * valid_mask).sum()
        # weight_sum = (weights * valid_mask).sum()
        # loss = weighted_loss_sum / (weight_sum + 1e-8)
        
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
        B, seq_len, _ = h.shape
        assert seq_len % self.N == 0, f"Sequence length {seq_len} is not a multiple of block size {self.N}"
        assert start_pos is not None, "Inference mode requires start_pos."
        assert time is not None, "Inference mode requires time."

        self.freqs_cis = self.freqs_cis.to(h.device)
        time_tensor = build_time_tensor(time, seq_len, B, self.N).to(h.device)
        t_vec = self.time_embed(timestep_embedding(time_tensor, self.time_dim))

        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]
        mask = build_inference_mask(start_pos, seq_len, self.N).to(h.device)

        xt = h[:, -self.N:, :].detach().clone()

        for layer in self.layers:
            h = layer(h, t_vec, start_pos, freqs_cis, mask)

        h = self.final_layer_norm(h)
        logits = self.output_proj(h)

        t_full = time_tensor[:, -1].reshape(-1, 1, 1)
        t_full = t_full.expand(-1, self.N, -1)  #

        flowlogits = self.compute_flow_logits(xt, t_full)
        logits[:, -self.N:, :].sub_(flowlogits)

        return {"logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        time_schedule: List[float],
        top_p: float = 1.0,
        echo: bool = False,
        pad_id: int = 0,
        eos_id: int = 1,
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
        B = len(prompt_tokens)
        assert B <= self.max_B, f"Batch size {B} exceeds maximum batch size {self.max_B}."
        assert all(0 <= x <= 1 for x in time_schedule), "Time steps must between 0 and 1."
        # assert time_schedule[0] == 0 and time_schedule[-1] == 1, "Time schedule must start with 0 and end with 1."

        max_prompt_len = max(len(t) for t in prompt_tokens)
        ctx_len = self.M * self.N
        assert max_prompt_len <= ctx_len, f"Prompt length {max_prompt_len} exceeds context length {ctx_len}."
        total_len = ctx_len

        tokens = torch.full((B, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, prompt in enumerate(prompt_tokens):
            prompt_len = len(prompt)
            # This is 0 if prompt_len is already a multiple of N.
            extra_pad = (-prompt_len) % self.N
            # prepad prompt tokens on the left to be multiples of N
            new_prompt = [pad_id] * extra_pad + prompt
            # Replace tokens tensor with prompt tokens
            tokens[k, :len(new_prompt)] = torch.tensor(new_prompt, dtype=torch.long, device="cuda")

        min_prompt_len = min(len(t) for t in prompt_tokens)
        min_padded_len = ((min_prompt_len) // self.N + 1) * self.N

        prev_pos = 0
        eos_reached = torch.tensor([False] * B, device="cuda")
        input_text_mask = tokens != pad_id

        import numpy as np
        Xt_storage = {}

        for cur_pos in range(min_padded_len, total_len, self.N):
            Xt = self.embed_for_inference(tokens[:, prev_pos:cur_pos])
            block_idx = cur_pos // self.N
            tracked_Xt = [] if block_idx in [50, 100, 150, 200] else None

            for i, (time, next_time) in enumerate(zip(time_schedule[:-1], time_schedule[1:])):
                logits = self.forward(Xt, start_pos=prev_pos, time=time)["logits"]
                # logits = self.forward(Xt, start_pos=0, time=time)["logits"]
                if i == 0:
                    prev_pos = cur_pos
                    Xt = Xt[:, -self.N:]

                probs = torch.softmax(logits[:, -self.N:], dim=-1)
                # probs = nucleus_cutoff(probs, top_p) if top_p < 1 else probs

                E = self.token_embed.weight
                X1t = torch.matmul(probs, E) # \Hat{X1} estimation at time t is exptectation of embedding vectors
                alpha = (next_time - time) / (1 - time) # (t_{i+1} - t_i) / (1 - t_i)
                Xt.lerp_(X1t, alpha)

                if tracked_Xt is not None:
                    tracked_Xt.append(Xt[0].detach().cpu().numpy())
            
            if tracked_Xt is not None:
                Xt_storage[block_idx] = np.stack(tracked_Xt, axis=0)
                print(f"Stored X_t evolution for block {block_idx}: shape {Xt_storage[block_idx].shape}")

            X1_logits = self.forward(Xt, start_pos=prev_pos, time=0.9)["logits"]

            probs  = F.softmax(X1_logits, dim=-1)         # (B, N, V)

            B, N, V = probs.shape
            probs_flat = probs.view(-1, V)             # (B*N, V)
            probs_flat = nucleus_cutoff(probs_flat, top_p) if top_p < 1 else probs_flat

            # draw one sample per row
            samples_flat = torch.multinomial(probs_flat, num_samples=1)  # (B*N, 1)

            # reshape back to (B, N)
            next_tokens = samples_flat.squeeze(-1).view(B, N)

            next_tokens = torch.where(input_text_mask[:, cur_pos:cur_pos+self.N], tokens[:,cur_pos:cur_pos+self.N], next_tokens)
            tokens[:, cur_pos:cur_pos+self.N] = next_tokens

            # Update the eos_reached flag for each sequence.
            eos_in_block = (
                (~input_text_mask[:, cur_pos:cur_pos+self.N]
                 ) & (next_tokens == eos_id)
            ).any(dim=1)
            eos_reached |= eos_in_block

            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            # start = 0 if echo else len(prompt_tokens[i])
            # toks = toks[start: ctx_len]
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx]
                
            out_tokens.append(toks)

        np.save('Xt_storage.npy', Xt_storage)
        print("Saved X_t evolution data to 'Xt_storage.npy'.")
        return out_tokens
