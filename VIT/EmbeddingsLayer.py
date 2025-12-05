import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np


class PatchEmbeddings(nn.Module):

    def __init__(self, image_size: int, d_model: int, patch_size, num_channels: int):

        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.linear_project = nn.Conv2d(
            in_channels=num_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        ).to(device=self.device)

    def forward(self, x: torch.Tensor):
        """
        x here is an image and here its differents dimensions
        B : the batch size
        C : number of channels
        H : height
        W : width
        Args:
            x (): torch.Tensor
        """

        # (B , C , H ,W) --> (B , d_model, N_Patch , N_Patch)
        x = self.linear_project(x)
        # (B , d_model, N_Patch , N_Patch) -> (B,d_model, N_Patch²) -> (B, N_patch², d_model)
        x = x.flatten(2).transpose(1, 2)

        return x


class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model, max_seq_length):

        super().__init__()

        # Create the CLS token that will contains all the info of the image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Create the positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        # We create a PE for every patch
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos][i] = np.sin(pos / (10000 ** ((i - 1) / d_model)))

        # Then we need to save the PE to the model but not as parameter that gonna be trained
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):

        # Expand the cls token to have one for every image of the batch
        cls_tokens = self.cls_token.expand(x.size()[0], -1, -1)

        # Concatenate these cls tokens to the embeddings
        x = torch.concat((cls_tokens, x), dim=1)

        # Add the PE to the embeddings
        x = x + self.pe

        return x
