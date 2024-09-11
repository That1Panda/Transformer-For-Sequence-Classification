import pytest
import torch
from transformers import AutoConfig

from encoder import Encoder  # Adjust based on your file structure


@pytest.mark.parametrize(
    "batch_size, seq_len, model_ckpt, vocab_size",
    [
        (2, 10, "bert-base-uncased", 30522),
        (4, 20, "bert-large-uncased", 256),
        (1, 5, "bert-base-uncased", 9999),
    ],
)
def test_encoder_output_size(batch_size, seq_len, model_ckpt, vocab_size):
    # Load the configuration from the specified pretrained checkpoint
    config = AutoConfig.from_pretrained(model_ckpt)

    # Override vocab_size in the config
    config.vocab_size = vocab_size

    # Initialize the Encoder with the loaded configuration
    encoder = Encoder(config)

    # Create a dummy input tensor of token IDs with shape (batch_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass through the encoder
    output = encoder(input_ids)

    # Assert that the output shape matches the expected shape: (batch_size, seq_len, hidden_size)
    assert output.shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Expected output shape {(batch_size, seq_len, config.hidden_size)} but got {output.shape}"
