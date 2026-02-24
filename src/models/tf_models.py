"""TensorFlow/Keras comparison models for orbit prediction.

Provides LSTM and Transformer implementations in Keras for
cross-framework benchmarking.
"""


def build_lstm_model(
    input_shape: tuple,
    horizon: int = 360,
    output_dim: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
):
    """Build a Keras LSTM model for orbit prediction.

    Args:
        input_shape: (seq_len, features) — e.g., (1440, 6)
        horizon: Number of output time steps
        output_dim: Features per output step
        hidden_dim: LSTM units
        num_layers: Number of LSTM layers
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("TensorFlow not available")
        return None

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=return_sequences, dropout=dropout)
        ))

    model.add(layers.Dense(hidden_dim, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(horizon * output_dim))
    model.add(layers.Reshape((horizon, output_dim)))

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_transformer_model(
    input_shape: tuple,
    horizon: int = 360,
    output_dim: int = 3,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.1,
):
    """Build a Keras Transformer (encoder-only) model.

    Args:
        input_shape: (seq_len, features)
        horizon: Number of output time steps
        output_dim: Features per output step
        d_model: Transformer dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        ff_dim: Feed-forward dimension
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("TensorFlow not available")
        return None

    inputs = layers.Input(shape=input_shape)

    # Project to model dimension
    x = layers.Dense(d_model)(inputs)

    # Add positional information via learned embeddings
    seq_len = input_shape[0]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)(positions)
    x = x + pos_embedding

    # Transformer encoder blocks
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization()(x + attn_output)

        # Feed-forward
        ff_output = layers.Dense(ff_dim, activation="gelu")(x)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        x = layers.LayerNormalization()(x + ff_output)

    # Global average pooling + output head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(horizon * output_dim)(x)
    outputs = layers.Reshape((horizon, output_dim))(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss="mse",
        metrics=["mae"],
    )

    return model
