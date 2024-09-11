import pytest
import torch
from transformers import AutoConfig

from encoder_layer import EncoderLayer


@pytest.mark.parametrize(
    "batch_size, seq_len, model_ckpt",
    [
        (2, 10, "bert-base-uncased"),  # Test case 1: BERT-base
        (4, 20, "bert-large-uncased"),  # Test case 2: BERT-large
        (1, 5, "bert-base-uncased"),  # Test case 3: Smaller input with BERT-base
    ],
)
def test_encoder_layer_output_size(batch_size, seq_len, model_ckpt):
    # Load the configuration from the specified pretrained checkpoint
    config = AutoConfig.from_pretrained(model_ckpt)

    # Initialize the EncoderLayer with the loaded configuration
    encoder_layer = EncoderLayer(config)

    # Create a dummy input tensor of shape (batch_size, seq_len, hidden_size)
    hidden_state = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass through the encoder layer
    output = encoder_layer(hidden_state)

    # Assert that the output shape matches the input shape: (batch_size, seq_len, hidden_size)
    assert output.shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Expected output shape {(batch_size, seq_len, config.hidden_size)} but got {output.shape}"
