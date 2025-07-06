from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F


def _build_supporters_for_packed_batch(input_ids: torch.Tensor, eos_token_id, nhead):
    B, L = input_ids.shape
    raw_idx = torch.arange(L, device=input_ids.device).expand_as(input_ids)
    last_eos_idx = torch.cummax(
        torch.where(input_ids == eos_token_id, raw_idx, 0),
        dim=1
    ).values
    relative_idx = raw_idx - last_eos_idx

    segment_id = (input_ids == eos_token_id).cumsum(dim=1)
    same_segment_mask = segment_id.unsqueeze(2) == segment_id.unsqueeze(1)
    causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=input_ids.device))
    should_attend_mask = same_segment_mask & causal_mask
    additive_attention_mask = torch.where(should_attend_mask, 0, -torch.inf)
    multihead_additive_attention_mask = additive_attention_mask.repeat_interleave(nhead, dim=0)

    return relative_idx, multihead_additive_attention_mask


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


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, dropout_p: float, device):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, device=device)
        self.k_proj = nn.Linear(d_model, d_model, device=device)
        self.v_proj = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        # x.shape = (batch_size, seq_len, d_model)
        # attn_maks.shape = (batch_size, seq_len, seq_len)
        weight_logits = self.q_proj(x) @ self.k_proj(x).transpose(1, 2) + attn_mask
        weights = self.dropout(F.softmax(weight_logits, dim=-1))
        return weights @ self.v_proj(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout_p: float, device):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model` must be a multiple of `num_heads`"
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(d_model=d_model // num_heads, dropout_p=dropout_p, device=device)
                for _ in range(num_heads)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model, device=device)

    def forward(self, x, attn_mask: torch.Tensor):
        head_results = torch.cat(
            [
                head(x, attn_mask)
                for head in self.attention_heads
            ],
            dim=-1
        )
        return self.out_proj(head_results)


class BasicLayersEncoderGPT2(nn.Module):
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
        nn.TransformerEncoderLayer
        self.transformer_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                nn.LayerNorm(d_model, device=device),
                                # TODO: Decompose this
                                nn.MultiheadAttention(
                                    embed_dim=d_model,
                                    num_heads=nhead,
                                    device=device,
                                    batch_first=True,
                                ),
                                nn.Dropout(p=0.1),
                            ]
                        ),
                        nn.ModuleList(
                            [
                                nn.LayerNorm(d_model, device=device),
                                nn.Linear(d_model, dim_feedforward, device=device),
                                nn.GELU(),
                                nn.Linear(dim_feedforward, d_model, device=device),
                                nn.Dropout(p=0.1),
                            ]
                        )
                    ]
                )
                for _ in range(num_layers)
            ]
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
        encoded = self.token_embedder(input_ids) * sqrt(self.d_model) + self.positional_embedder(input_idx)
        for transformer_layer in self.transformer_layers:
            for subblock in transformer_layer:
                x = encoded
                for layer in subblock:
                    if isinstance(layer, nn.MultiheadAttention):
                        x = layer(
                            x,
                            x,
                            x,
                            need_weights=False,
                            attn_mask=mask,
                        )[0]
                    else:
                        x = layer(x)
                encoded = encoded + x
        logits = self.decoder(encoded)
        return logits
