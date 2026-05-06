"""
Phishing URL Classification - CNN Baseline Model

Architecture (from plan):
- Character/byte embedding
- Parallel multi-scale convolutions (k=3, 5, 7)
- Stacked conv blocks with pooling
- Global max pool → MLP → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phishguard.data import PAD_IDX, VOCAB_SIZE


class PhishingCNN(nn.Module):
    """TextCNN-style model for phishing URL classification.

    Architecture:
        Input: (batch, max_len) byte indices
        Embedding: (batch, max_len, embed_dim)
        Parallel convs: k=3,5,7 with 128 filters each → concat → (batch, max_len, 384)
        Conv block 1: k=3, 256 filters → pool → (batch, max_len//2, 256)
        Conv block 2: k=3, 128 filters → pool → (batch, max_len//4, 128)
        Global max pool → (batch, 128)
        MLP: 256 → dropout → 1 → sigmoid
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.3,
        padding_idx: int = PAD_IDX,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # Parallel multi-scale convolutions
        self.parallel_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_filters),
                )
                for k in kernel_sizes
            ]
        )

        # After concat: num_filters * len(kernel_sizes) channels
        concat_dim = num_filters * len(kernel_sizes)  # 384

        # Stacked conv blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(concat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, max_len) tensor of byte indices

        Returns:
            logits: (batch, 1) tensor of logits (apply sigmoid for probabilities)
        """
        # Embedding: (batch, max_len) → (batch, max_len, embed_dim)
        x = self.embedding(input_ids)

        # Conv1d expects (batch, channels, length), so transpose
        x = x.transpose(1, 2)  # (batch, embed_dim, max_len)

        # Parallel convolutions
        conv_outputs = [conv(x) for conv in self.parallel_convs]
        x = torch.cat(conv_outputs, dim=1)  # (batch, 384, max_len)

        # Stacked conv blocks
        x = self.conv_block1(x)  # (batch, 256, max_len//2)
        x = self.conv_block2(x)  # (batch, 128, max_len//4)

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, 128)

        # Classification
        logits = self.classifier(x)  # (batch, 1)

        return logits

    def get_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract features before global pooling (for transformer input).

        Args:
            input_ids: (batch, max_len) tensor of byte indices

        Returns:
            features: (batch, max_len//4, 128) tensor
        """
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        conv_outputs = [conv(x) for conv in self.parallel_convs]
        x = torch.cat(conv_outputs, dim=1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Transpose back: (batch, 128, seq_len) → (batch, seq_len, 128)
        x = x.transpose(1, 2)

        return x
