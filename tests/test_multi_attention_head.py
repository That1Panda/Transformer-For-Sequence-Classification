import torch
import pytest
from transformers import AutoConfig
from multi_head_attention import (
    MultiHeadAttention,
)


@pytest.mark.parametrize(
    "batch_size, seq_len, model_ckpt",
    [(2, 10, "bert-base-uncased"), (4, 20, "distilbert-base-uncased")],
)
def test_multi_head_attention_output_size(batch_size, seq_len, model_ckpt):
    # Load the configuration from a pretrained checkpoint
    config = AutoConfig.from_pretrained(model_ckpt)

    # Initialize the MultiHeadAttention with the loaded configuration
    multi_head_attention = MultiHeadAttention(config)

    # Create a dummy input tensor of shape (batch_size, seq_len, hidden_size)
    hidden_state = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass through the multi-head attention
    output = multi_head_attention(hidden_state)

    # Assert that the output shape is correct
    assert output.shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Expected output shape {(batch_size, seq_len, config.hidden_size)} but got {output.shape}"
