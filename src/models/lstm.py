"""LSTM model for orbit prediction.

Bidirectional LSTM encoder-decoder architecture that maps input sequences
of spacecraft positions/velocities to predicted future positions.
"""

import torch
import torch.nn as nn


class OrbitLSTM(nn.Module):
    """Bidirectional LSTM for orbit time-series prediction.

    Architecture:
        Input (batch, seq_len, input_dim)
        -> Bidirectional LSTM encoder (2 layers)
        -> LSTM decoder (2 layers)
        -> Linear projection
        -> Output (batch, horizon, 3)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input features (default 6: x,y,z + vx,vy,vz)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            horizon: Number of output time steps
            output_dim: Output features per step (default 3: x,y,z)
            dropout: Dropout rate between layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.output_dim = output_dim

        # Encoder: bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bridge: project bidirectional hidden to decoder hidden
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder: unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Learnable start token for decoder
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) input sequence

        Returns:
            (batch, horizon, output_dim) predicted positions
        """
        batch_size = x.size(0)

        # Encode
        _, (h_enc, c_enc) = self.encoder(x)

        # Reshape bidirectional hidden states: (num_layers*2, batch, hidden)
        # -> merge directions: (num_layers, batch, hidden*2)
        h_enc = h_enc.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_enc = torch.cat([h_enc[:, 0], h_enc[:, 1]], dim=-1)
        c_enc = c_enc.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c_enc = torch.cat([c_enc[:, 0], c_enc[:, 1]], dim=-1)

        # Bridge to decoder dimension
        h_dec = self.bridge_h(h_enc)
        c_dec = self.bridge_c(c_enc)

        # Decode autoregressively
        decoder_input = self.start_token.expand(batch_size, -1, -1)
        outputs = []

        for _ in range(self.horizon):
            dec_out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            pred = self.output_proj(dec_out)
            outputs.append(pred)
            decoder_input = pred  # Feed prediction as next input

        return torch.cat(outputs, dim=1)


class OrbitLSTMDirect(nn.Module):
    """Simpler LSTM that directly outputs the full horizon at once.

    Faster to train than autoregressive version, but may be less accurate
    for very long horizons.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, horizon, output_dim)
        """
        _, (h, _) = self.lstm(x)

        # Merge bidirectional final hidden states from last layer
        h = torch.cat([h[-2], h[-1]], dim=-1)
        out = self.fc(h)
        return out.view(-1, self.horizon, self.output_dim)
