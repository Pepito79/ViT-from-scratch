import torch
import torch.nn as nn
from AttentionLayer import MultiHeadAttention


class Encoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int):

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Define the Multi head attention
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        # We need two "add and norm" layers , the argument is not the index of the dim but the dim itself
        self.normalization_1 = nn.LayerNorm(d_model)
        self.normalization_2 = nn.LayerNorm(d_model)

        # FFN block which aims to project the embeddings in a higher dimension and then to re-project it in the initial
        # dimension to capture more complexe patterns
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor):

        # We use post-normalisation for better stability
        out = x + self.mha(self.normalization_1(x))
        out = out + self.ffn(self.normalization_2(out))

        # We finally have a better representation of our initial embeddings
        return out
