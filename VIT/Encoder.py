import torch
import torch.nn as nn
from AttentionLayer import MultiHeadAttention


class Encoder(nn.Module):

    def __init__(self, d_model: int, n_head: int):

        super().__init__()

        self.d_model = d_model
        self.n_head = n_head

        # Define the Multi head attention
        self.mha = MultiHeadAttention(
            d_model= 
        )
