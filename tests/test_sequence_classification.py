import pytest
import torch
from transformers import AutoConfig

from sequence_classification import SequenceClassification


@pytest.mark.parametrize(
    "batch_size, seq_len, num_labels, model_ckpt, vocab_size",
    [
        (2, 10, 3, "bert-base-uncased", 30522),
        (4, 20, 2, "bert-large-uncased", 256),
        (1, 5, 5, "bert-base-uncased", 9999),
    ],
)
def test_sequence_classification_output_size(
    batch_size, seq_len, num_labels, model_ckpt, vocab_size
):
    # Load the configuration from the specified pretrained checkpoint
    config = AutoConfig.from_pretrained(model_ckpt)

    # Override specific attributes in the config
    config.num_labels = num_labels
    config.vocab_size = vocab_size

    # Initialize the SequenceClassification model with the loaded configuration
    model = SequenceClassification(config)

    # Create a dummy input tensor of token IDs with shape (batch_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass through the model
    output = model(input_ids)

    # Assert that the output shape matches the expected shape: (batch_size, num_labels)
    assert output.shape == (
        batch_size,
        num_labels,
    ), f"Expected output shape {(batch_size, num_labels)} but got {output.shape}"
