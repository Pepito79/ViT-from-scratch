import torch
import torch.nn as nn
from EmbeddingsLayer import PatchEmbeddings, PositionalEmbeddings
from AttentionLayer import MultiHeadAttention
from Encoder import Encoder


class ViT(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        n_heads,
        d_model,
        n_channels,
        n_layers,
        n_classes,
    ):

        super().__init__()

        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (
            self.patch_size[0] * self.patch_size[1]
        )

        # We need to add the CLS token
        self.max_seq_length = self.n_patches + 1

        # We generate the embeddings of the patches
        self.patch_embeddings = PatchEmbeddings(
            image_size=img_size,
            d_model=d_model,
            patch_size=patch_size,
            num_channels=n_channels,
        )

        # We generate the Postionnal encodings
        self.pe = PositionalEmbeddings(
            d_model=d_model, max_seq_length=self.max_seq_length
        )

        # The * allows us to unpack the list
        self.encoder_layers = nn.Sequential(
            *[Encoder(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )

        # Finally we classify the CLS token
        self.classifier = nn.Sequential(
            nn.Linear(d_model, n_classes), nn.Softmax(dim=-1)
        )

    def forward(self, images):
        x = self.patch_embeddings(images)
        x = self.pe(x)
        x = self.encoder_layers(x)

        # Only classify the [CLS] token
        x = self.classifier(x[:, 0])
        return x
