from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

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
        device,
    ):
        super().__init__()
        self.d_model = d_model
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
            if getattr(m, "in_proj_bias", None) is not None:
                nn.init.zeros_(m.in_proj_bias)
            # `m.out_proj` is an instance of `nn.Linear`, thus already handled by the first condition

    def forward(self, input_ids: torch.Tensor):
        input_idx = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand_as(input_ids)
        embedded = self.token_embedder(input_ids) * sqrt(self.d_model) + self.positional_embedder(input_idx)
        transformed = self.transformer(
            embedded,
            mask=nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1], device=input_ids.device),
        )
        logits = self.decoder(transformed)
        return logits
