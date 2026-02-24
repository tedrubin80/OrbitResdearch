"""Transformer model for orbit prediction.

Encoder-decoder Transformer with sinusoidal positional encoding,
designed for spacecraft trajectory time-series forecasting.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series transformers."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class OrbitTransformer(nn.Module):
    """Transformer encoder-decoder for orbit prediction.

    Architecture:
        Input -> Linear projection -> Positional Encoding
        -> Transformer Encoder (N layers, multi-head attention)
        -> Transformer Decoder (N layers, cross-attention)
        -> Linear projection -> Output
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feed-forward network dimension
            horizon: Number of prediction steps
            output_dim: Output features per step (3 for x,y,z)
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.horizon = horizon
        self.output_dim = output_dim

        # Input projections
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj_in = nn.Linear(output_dim, d_model)

        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        # Learnable query tokens for decoder
        self.query_tokens = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) input sequence

        Returns:
            (batch, horizon, output_dim) predicted positions
        """
        batch_size = x.size(0)

        # Encode input sequence
        src = self.input_proj(x) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        # Decode with learnable query tokens
        tgt = self.query_tokens.expand(batch_size, -1, -1)
        tgt = self.pos_decoder(tgt)

        # Generate causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.horizon, device=x.device
        )

        decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output_proj(decoded)

        return output


class OrbitTransformerDirect(nn.Module):
    """Encoder-only Transformer that directly outputs full horizon.

    Simpler alternative — uses only the encoder with a global pooling
    and MLP head to predict the full output sequence at once.
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, horizon, output_dim)
        """
        src = self.input_proj(x) * math.sqrt(src.size(-1)) if False else self.input_proj(x)
        src = self.pos_encoding(src)
        encoded = self.encoder(src)

        # Global average pooling over sequence dimension
        pooled = encoded.mean(dim=1)
        out = self.head(pooled)
        return out.view(-1, self.horizon, self.output_dim)
