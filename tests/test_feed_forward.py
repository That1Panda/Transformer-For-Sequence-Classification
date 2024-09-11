import torch
import pytest
from transformers import AutoConfig
from feed_forward import FeedForward  # Adjust based on your file structure


@pytest.mark.parametrize(
    "batch_size, seq_len, model_ckpt",
    [(2, 10, "bert-base-uncased"), (4, 20, "distilbert-base-uncased")],
)
def test_feed_forward_output_size(batch_size, seq_len, model_ckpt):
    # Load the configuration from the specified pretrained checkpoint
    config = AutoConfig.from_pretrained(model_ckpt)

    # Initialize the FeedForward layer with the loaded configuration
    feed_forward = FeedForward(config)

    # Create a dummy input tensor with shape (batch_size, seq_len, hidden_size)
    hidden_state = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass through the feed-forward network
    output = feed_forward(hidden_state)

    # Assert that the output shape matches the input shape: (batch_size, seq_len, hidden_size)
    assert output.shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Expected output shape {(batch_size, seq_len, config.hidden_size)} but got {output.shape}"
