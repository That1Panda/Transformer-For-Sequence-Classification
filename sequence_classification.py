import torch
import torch.nn as nn
from transformers import AutoConfig

from encoder import Encoder


class SequenceClassification(nn.Module):
    """
    Implements a sequence classification model using a Transformer encoder and a classification head.

    Args:
        config (AutoConfig): A configuration object from the Hugging Face transformers library that provides:
            - hidden_size (int): Dimensionality of the hidden states.
            - num_labels (int): Number of labels for classification.
            - hidden_dropout_prob (float): Dropout probability for the dropout layer.
    """

    def __init__(self, config: AutoConfig):
        super().__init__()

        # Transformer encoder
        self.encoder = Encoder(config)

        # Classification head (linear layer to map hidden states to label space)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence classification.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor containing logits of shape (batch_size, num_labels).
        """
        # Pass input through the encoder, take the output corresponding to the [CLS] token
        x = self.encoder(input_ids)[:, 0, :]

        # Apply dropout
        x = self.dropout(x)

        # Apply the classification layer
        x = self.classifier(x)

        return x
