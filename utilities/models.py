from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _build_supporters_for_packed_batch(input_ids: torch.Tensor, eos_token_id: int, nhead: int):
    _, seq_len = input_ids.shape
    raw_idx = torch.arange(seq_len, device=input_ids.device).expand_as(input_ids)
    last_eos_idx = torch.cummax(
        torch.where(input_ids == eos_token_id, raw_idx, 0),
        dim=1
    ).values
    relative_idx = raw_idx - last_eos_idx  # shape: (batch_size, seq_len)

    segment_id = (input_ids == eos_token_id).cumsum(dim=1)
    same_segment_mask = segment_id.unsqueeze(2) == segment_id.unsqueeze(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
    should_attend_mask = same_segment_mask & causal_mask
    additive_attention_mask = torch.where(should_attend_mask, 0, -torch.inf).bfloat16()
    multihead_additive_attention_mask = additive_attention_mask.repeat_interleave(
        repeats=nhead, dim=0
    )  # shape: (batch_size * nhead, seq_len, seq_len)

    return relative_idx, multihead_additive_attention_mask


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in, dtype=torch.bfloat16, device=device))
        self.bias = nn.Parameter(torch.zeros(d_out, dtype=torch.bfloat16, device=device)) if bias else None

    def forward(self, x: torch.Tensor):
        x = x.bfloat16()
        # x.shape = (batch_size, seq_len, d_in)
        # x.unsqueeze(-1).shape = (batch_size, seq_len, d_in, 1)
        # self.weight.shape = (d_out, d_in)
        # (self.weight @ x.unsqueeze(-1)).shape = (batch_size, seq_len, d_out, 1)
        y = (self.weight @ x.unsqueeze(-1)).squeeze(-1)
        # y.shape = (batch_size, seq_len, d_out)
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor):
        # x.shape = (batch_size, seq_len)
        return self.weight[x]


class LayerNorm(nn.Module):
    def __init__(self, d_layer: int, device, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_layer, device=device))
        self.bias = nn.Parameter(torch.zeros(d_layer, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x = x.float()
        # x.shape = (batch_size, seq_len, d_layer)
        normalized = (
            (x - x.mean(dim=-1, keepdim=True))
            / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + self.eps)
        )
        # normalized.shape = (batch_size, seq_len, d_layer)
        denormalized = self.weight * normalized + self.bias
        # denormalized.shape = (batch_size, seq_len, d_layer)
        return denormalized


class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.float()
        # x.shape = (batch_size, seq_len, d_layer)
        return x * 0.5 * (1 + torch.erf(x / sqrt(2)))


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p <= 1, f"p must be in [0, 1], but had value {p}"
        self.p = p

    def forward(self, x: torch.Tensor):
        x = x.bfloat16()
        # x.shape = (batch_size, seq_len, d_layer)
        if not self.training or self.p == 0:
            return x
        gate = (torch.rand_like(x, dtype=torch.bfloat16) > self.p).bfloat16()
        # gate.shape = (batch_size, seq_len, d_layer)
        return (gate * x) / (1. - self.p)


class Softmax(nn.Module):
    def forward(self, x: torch.Tensor, dim=-1):
        x = x.float()
        shift_x = x - x.max(dim=dim, keepdim=True).values
        shift_exp = shift_x.exp()
        return shift_exp / shift_exp.sum(dim=dim, keepdim=True)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float, device):
        super().__init__()
        d_head = d_model // num_heads
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self.q_proj = Linear(d_model, d_head, device=device)
        self.k_proj = Linear(d_model, d_head, device=device)
        self.v_proj = Linear(d_model, d_head, device=device)
        self.dropout = Dropout(p=dropout_p)
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, use_kv_cache: bool):
        # x.shape = (batch_size, seq_len | 1, d_model)
        # attn_maks.shape = (batch_size, seq_len, seq_len)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if use_kv_cache and self.kv_cache is not None:
            # kv_cache[0].shape = (batch_size, seq_len - 1, d_head)
            k = torch.cat((self.kv_cache[0], k), dim=1)
            v = torch.cat((self.kv_cache[1], v), dim=1)
        self.kv_cache = (k, v)

        weight_logits = q @ k.transpose(1, 2) + attn_mask
        weights = self.dropout(self.softmax(weight_logits))
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout_p: float, device):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model` must be a multiple of `num_heads`"
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, device=device)
                for _ in range(num_heads)
            ]
        )
        self.out_proj = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, use_kv_cache: bool):
        batch_size = x.shape[0]
        head_results = torch.cat(
            [
                head(x, attn_mask[i * batch_size : (i + 1) * batch_size], use_kv_cache)
                for i, head in enumerate(self.attention_heads)
            ],
            dim=-1
        )
        return self.out_proj(head_results)


class FasterMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout_p: float, device):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model` must be a multiple of `num_heads`"
        self.num_heads = num_heads
        self.d_model = d_model
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self.in_proj = Linear(d_model, d_model * 3, device=device)
        self.softmax = Softmax()
        self.dropout = Dropout(p=dropout_p)
        self.out_proj = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, use_kv_cache: bool):
        # x.shape = (batch_size, seq_len | 1, d_model)
        batch_size = x.shape[0]
        out_seq_len = x.shape[1]
        d_head = self.d_model // self.num_heads

        projected = self.in_proj(x)  # shape: (batch_size, out_seq_len, d_model * 3)
        qkv = (
            projected
            .view(batch_size, out_seq_len, self.num_heads, d_head, 3)
            .transpose(1, 2)
            .reshape(batch_size * self.num_heads, out_seq_len, d_head, 3)
        )
        q = qkv[:, :, :, 0]  # shape: (batch_size * num_heads, out_seq_len, d_head)
        k = qkv[:, :, :, 1]  # shape: (batch_size * num_heads, out_seq_len, d_head)
        v = qkv[:, :, :, 2]  # shape: (batch_size * num_heads, out_seq_len, d_head)

        if use_kv_cache and self.kv_cache is not None:
            # kv_cache[0].shape = (batch_size, seq_len - 1, d_head)
            k = torch.cat((self.kv_cache[0], k), dim=1)  # shape: (batch_size * num_heads, seq_len, d_head)
            v = torch.cat((self.kv_cache[1], v), dim=1)  # shape: (batch_size * num_heads, seq_len, d_head)
        self.kv_cache = (k, v)

        # Surprisingly, this manual implementation ran faster than F.scaled_dot_product_attention here (on H100)
        weight_logits = (
            q @ k.transpose(-2, -1) / np.sqrt(d_head) + attn_mask
        )  # shape: (batch_size * num_heads, seq_len, seq_len)
        weights = self.dropout(self.softmax(weight_logits))
        raw_head_results = weights @ v  # shape: (batch_size * num_heads, seq_len, d_head)
        head_results = (
            raw_head_results
            .view(batch_size, self.num_heads, out_seq_len, d_head)
            .transpose(1, 2)
            .reshape(batch_size, out_seq_len, self.d_model)
        )

        return self.out_proj(head_results)


@dataclass
class ModelConfig:
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    vocab_size: int
    context_length: int
    eos_token_id: int
    dropout_p: float
    device: str | torch.device


class ParametersGPT2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedder = Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model, device=config.device
        )
        self.positional_embedder = Embedding(
            num_embeddings=config.context_length, embedding_dim=config.d_model, device=config.device
        )
        self.transformer_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                LayerNorm(config.d_model, device=config.device),
                                FasterMultiHeadAttention(
                                    d_model=config.d_model,
                                    num_heads=config.nhead,
                                    dropout_p=config.dropout_p,
                                    device=config.device,
                                ),
                                Dropout(p=config.dropout_p),
                            ]
                        ),
                        nn.ModuleList(
                            [
                                LayerNorm(config.d_model, device=config.device),
                                Linear(config.d_model, config.dim_feedforward, device=config.device),
                                GELU(),
                                Linear(config.dim_feedforward, config.d_model, device=config.device),
                                Dropout(p=config.dropout_p),
                            ]
                        )
                    ]
                )
                for _ in range(config.num_layers)
            ]
        )
        self.apply(self.init_weights)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False, device=config.device)
        self.decoder.weight = self.token_embedder.weight

    def init_weights(self, m):
        if isinstance(m, (Linear, Embedding)):
            std = 0.02
            if isinstance(m, Linear) and m.weight.shape[0] == self.config.d_model:
                std /= sqrt(self.config.num_layers)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, streaming: bool = False, seq_len: int | None = None):
        if streaming:
            assert seq_len is not None, "seq_len must be passed when streaming"
            input_idx = torch.ones_like(input_ids, device=input_ids.device) * seq_len
            mask = torch.zeros(self.config.nhead, 1, seq_len, dtype=torch.bfloat16, device=input_ids.device)
        else:
            input_idx, mask = _build_supporters_for_packed_batch(
                input_ids, eos_token_id=self.config.eos_token_id, nhead=self.config.nhead
            )

        encoded = self.token_embedder(input_ids) * sqrt(self.config.d_model) + self.positional_embedder(input_idx)
        for transformer_layer in self.transformer_layers:
            for subblock in transformer_layer:
                x = encoded
                for layer in subblock:
                    if isinstance(layer, FasterMultiHeadAttention):
                        x = layer(x, attn_mask=mask, use_kv_cache=streaming)
                    else:
                        x = layer(x)
                encoded = encoded + x
        logits = self.decoder(encoded)
        return logits


# --------------
# DEPRECATED
# --------------

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.numerators = 10_000 ** (
            torch.arange(
                start=0,
                end=embedding_dim,
                step=2,
            ).float()
            / embedding_dim
        )

    def forward(self, input_ids: torch.Tensor):
        with torch.no_grad():
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ).float()
            numerators = self.numerators.to(input_ids.device)
            raw_embeddings = positions.unsqueeze(1) @ (1 / numerators).unsqueeze(0)
            even_embeddings = torch.sin(raw_embeddings)
            odd_embeddings = torch.cos(raw_embeddings)
            embeddings = torch.stack(
                [even_embeddings, odd_embeddings], dim=-1
            ).view(
                len(positions), -1
            )
            return embeddings.unsqueeze(0).expand(input_ids.shape[0], -1, -1)


class TransformerEncoderGPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        context_length: int,
        eos_token_id: int,
        device,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.context_length = context_length
        self.eos_token_id = eos_token_id
        self.device = device
        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.positional_embedder = nn.Embedding(
            num_embeddings=context_length, embedding_dim=d_model, device=device
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                device=device,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_layers,
        )
        self.apply(self.init_weights)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False, device=device)
        self.decoder.weight = self.token_embedder.weight

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.in_proj_bias)
            # `m.out_proj` is an instance of `nn.Linear`, thus already handled by the first condition

    def forward(self, input_ids: torch.Tensor):
        input_idx, mask = _build_supporters_for_packed_batch(input_ids, eos_token_id=self.eos_token_id, nhead=self.nhead)
        embedded = self.token_embedder(input_ids) * sqrt(self.d_model) + self.positional_embedder(input_idx)
        transformed = self.transformer(
            embedded,
            mask=mask,
        )
        logits = self.decoder(transformed)
        return logits


class TransformerEncoderGPT2(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        context_length: int,
        eos_token_id: int,
        device,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.context_length = context_length
        self.eos_token_id = eos_token_id
        self.device = device
        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.positional_embedder = nn.Embedding(
            num_embeddings=context_length, embedding_dim=d_model, device=device
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                device=device,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.apply(self.init_weights)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False, device=device)
        self.decoder.weight = self.token_embedder.weight

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            std = 0.02
            if isinstance(m, nn.Linear) and m.weight.shape[0] == self.d_model:
                std /= sqrt(self.num_layers)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.in_proj_bias)
            # `m.out_proj` is an instance of `nn.Linear`, thus already handled by the first condition

    def forward(self, input_ids: torch.Tensor):
        input_idx, mask = _build_supporters_for_packed_batch(input_ids, eos_token_id=self.eos_token_id, nhead=self.nhead)
        embedded = self.token_embedder(input_ids) * sqrt(self.d_model) + self.positional_embedder(input_idx)
        transformed = self.transformer(
            embedded,
            mask=mask,
        )
        logits = self.decoder(transformed)
        return logits
