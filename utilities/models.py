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
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, vocab_size: int, device):
        super().__init__()
        self.device = device
        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.positional_embedder = PositionalEmbedding(embedding_dim=d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                device=device,
                batch_first=False,  # TODO: Try flipping this to True and removing permute calls
                norm_first=False,
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(d_model, vocab_size, bias=False, device=device)
        self.decoder.weight = self.token_embedder.weight

    def forward(self, input_ids: torch.Tensor):
        embedded = self.token_embedder(input_ids) + self.positional_embedder(input_ids)
        transformed = self.transformer(
            embedded.permute(1, 0, 2),  # Transformer expects (seq_len, batch_size, features)
            mask=nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1], device=input_ids.device),
        )
        logits = self.decoder(transformed.permute(1, 0, 2))
        return logits
