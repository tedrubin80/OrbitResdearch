"""Multi-modal model combining orbit positions with solar wind data.

Dual-encoder architecture with attention-based fusion for predicting
orbit perturbations during geomagnetic storms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Attention layer that lets one modality attend to another."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model) — primary modality
            context: (batch, seq_len, d_model) — modality to attend to

        Returns:
            (batch, seq_len, d_model) — attended query
        """
        attended, _ = self.attn(query, context, context)
        return self.norm(query + attended)


class SolarWindOrbitModel(nn.Module):
    """Dual-encoder model with attention-based fusion.

    Architecture:
        Orbit LSTM encoder → orbit features
        Solar wind LSTM encoder → solar features
        Cross-modal attention (orbit attends to solar wind)
        Fusion → prediction head
        Output: predicted position residuals

    The model predicts residuals (deviations from nominal orbit),
    not absolute positions. This focuses learning on the solar wind effect.
    """

    def __init__(
        self,
        orbit_input_dim: int = 6,
        solar_input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            orbit_input_dim: Orbit features (x,y,z,vx,vy,vz)
            solar_input_dim: Solar wind features (Bx,By,Bz,speed,density,Kp,Dst)
            hidden_dim: Encoder hidden dimension
            num_layers: Number of LSTM layers
            nhead: Attention heads for cross-modal fusion
            horizon: Prediction steps
            output_dim: Output features (x,y,z)
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.output_dim = output_dim

        # Orbit encoder
        self.orbit_proj = nn.Linear(orbit_input_dim, hidden_dim)
        self.orbit_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.orbit_norm = nn.LayerNorm(hidden_dim * 2)

        # Solar wind encoder
        self.solar_proj = nn.Linear(solar_input_dim, hidden_dim)
        self.solar_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.solar_norm = nn.LayerNorm(hidden_dim * 2)

        # Cross-modal attention: orbit attends to solar wind
        self.cross_attention = CrossModalAttention(
            d_model=hidden_dim * 2,
            nhead=nhead,
            dropout=dropout,
        )

        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(
        self,
        orbit_input: torch.Tensor,
        solar_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            orbit_input: (batch, seq_len, orbit_input_dim)
            solar_input: (batch, seq_len, solar_input_dim)

        Returns:
            (batch, horizon, output_dim) — predicted position residuals
        """
        # Encode orbit
        orbit_emb = self.orbit_proj(orbit_input)
        orbit_encoded, _ = self.orbit_encoder(orbit_emb)
        orbit_encoded = self.orbit_norm(orbit_encoded)

        # Encode solar wind
        solar_emb = self.solar_proj(solar_input)
        solar_encoded, _ = self.solar_encoder(solar_emb)
        solar_encoded = self.solar_norm(solar_encoded)

        # Cross-modal attention: orbit attends to solar wind patterns
        orbit_attended = self.cross_attention(orbit_encoded, solar_encoded)

        # Combine attended orbit features with solar context
        # Pool over sequence dimension
        orbit_pooled = orbit_attended.mean(dim=1)   # (batch, hidden*2)
        solar_pooled = solar_encoded.mean(dim=1)    # (batch, hidden*2)

        # Concatenate and fuse
        fused = torch.cat([orbit_pooled, solar_pooled], dim=-1)  # (batch, hidden*4)
        fused = self.fusion(fused)

        # Predict residuals
        out = self.prediction_head(fused)
        return out.view(-1, self.horizon, self.output_dim)


class SolarWindClassifier(nn.Module):
    """Binary classifier for detecting geomagnetic storm impacts.

    Predicts whether a spacecraft's orbit will be significantly
    perturbed (above threshold) given current solar wind conditions.
    Useful as a complementary model to the regression approach.
    """

    def __init__(
        self,
        solar_input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=solar_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solar_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            solar_input: (batch, seq_len, solar_input_dim)

        Returns:
            (batch, 1) — probability of significant perturbation
        """
        _, (h, _) = self.encoder(solar_input)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return torch.sigmoid(self.classifier(h))
