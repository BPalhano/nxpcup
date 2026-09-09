"""MicroViT Architecture for NXP Cup Autonomous Driving.

Combines a lightweight 2-stage convolutional stem for local feature extraction
with a Multi-Head Self-Attention Transformer encoder for global spatial correlation
between track boundaries and track center.

Target parameter budget: 350k - 450k parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Clean multi-head self-attention module optimized for ONNX and TensorRT export."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm architecture."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroViT(nn.Module):
    """MicroViT: Hybrid Lightweight Convolutional Stem + Transformer Encoder.

    Designed specifically for edge inference on scale 1:10 autonomous vehicles.
    Processes (B, 3, 66, 200) inputs, captures track features at both boundaries
    via global self-attention, and regresses continuous steering commands.
    """

    def __init__(
        self,
        input_channels: int = 3,
        stem_channels: tuple[int, int] = (32, 64),
        patch_size: tuple[int, int] = (2, 2),
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        output_dim: int = 1,
        use_tanh: bool = False,
    ) -> None:
        """Initialize MicroViT.

        Args:
            input_channels: Input image channels (3 for RGB).
            stem_channels: Channels for the 2-stage conv stem (default: (32, 64)).
            patch_size: Kernel size and stride for patch projection on stem output (default: (2, 2)).
            embed_dim: Token embedding dimension (default: 128).
            depth: Number of Transformer encoder blocks (default: 2, ~350k-390k params).
            num_heads: Number of attention heads (default: 4).
            mlp_ratio: Hidden dimension multiplier in MLP (default: 2.0 -> dim 256).
            dropout: Dropout probability.
            attention_dropout: Attention map dropout probability.
            output_dim: Output regression dimension (1 for steering angle).
            use_tanh: Whether to apply tanh activation to bound output in [-1, 1].
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_tanh = use_tanh

        # 1. Lightweight 2-stage Conv Stem:
        # (B, 3, 66, 200) -> Conv1 s=2 -> (B, 32, 33, 100) -> Conv2 s=2 -> (B, 64, 16, 50)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, stem_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_channels[0]),
            nn.GELU(),
            nn.Conv2d(
                stem_channels[0],
                stem_channels[1],
                kernel_size=3,
                stride=2,
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels[1]),
            nn.GELU(),
        )

        # 2. Patch Projection over reduced feature map (16, 50):
        # With patch_size=(2, 2), spatial becomes (16/2, 50/2) = (8, 25) -> 200 tokens
        self.patch_proj = nn.Conv2d(
            stem_channels[1],
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (16 // patch_size[0]) * (50 // patch_size[1])
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # 4. Regression Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated normal for positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 66, 200).

        Returns:
            Steering regression output of shape (B, 1).
        """
        # Stem feature extraction
        x = self.stem(x)  # (B, 64, 16, 50)

        # Patch embedding
        x = self.patch_proj(x)  # (B, embed_dim, 8, 25)
        x = x.flatten(2).transpose(1, 2)  # (B, 200, embed_dim)

        # Positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global Average Pooling across tokens
        x = x.mean(dim=1)  # (B, embed_dim)

        out = self.head(x)  # (B, 1)

        if self.use_tanh:
            out = torch.tanh(out)

        return out
