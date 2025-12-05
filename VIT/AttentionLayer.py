import torch
import torch.nn as nn


class AttentionHead(nn.Module):

    def __init__(self, d_model: int, head_size: int):
        super().__init__()

        self.d_model = d_model
        self.head_size = head_size

        # Q(i) = X@W_Q(i): it's a simple linear layer without bias
        self.query = nn.Linear(self.d_model, self.head_size, bias=False)

        # Same things for the others
        self.key = nn.Linear(self.d_model, self.head_size, bias=False)
        self.values = nn.Linear(self.d_model, self.head_size, bias=False)

    def forward(self, x: torch.Tensor):

        # Let's calculate our Q,K,V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.values(x)

        # Compute the attention mechanism
        temp = Q @ K.transpose(-2, -1)  # We need to transpose the matrix
        temp = temp / (self.head_size) ** 0.5  # We scale it like in the paper
        temp = torch.softmax(temp, dim=-1)  # We apply the softmax

        # And here we got our attention matrix
        attention = temp @ V

        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if d_model % n_heads != 0:
            raise Exception("d_model is not divible by yout n_heads")

        self.head_size = d_model // n_heads

        # Create the W_O weights matrix
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )

    def forward(self, x: torch.Tensor):

        # Concatenate the output of every head
        h = torch.cat([head(x) for head in self.heads], dim=-1)

        attention = self.W_O(h)

        return attention
