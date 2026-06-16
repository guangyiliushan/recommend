"""Model configuration Hydra YAML files.

Each model family has its own directory with per-model config YAML:
    - positional hyperparams (embedding_dim, hidden_dims, num_layers, ...)
    - training-specific (learning_rate, weight_decay, ...)
    - architecture-specific (num_heads, dropout, ...)

Override via Hydra CLI:
    python run.py model=deepfm model.params.embedding_dim=32
"""
