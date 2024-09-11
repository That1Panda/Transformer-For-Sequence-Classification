import torch
import torch.nn as nn
from transformers import AutoConfig

from embedding import Embedding
from encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Implements the Transformer encoder consisting of an embedding layer followed by multiple encoder layers.

    Args:
        config (AutoConfig): A configuration object from the Hugging Face transformers library that provides:
            - vocab_size (int): Vocabulary size of the model.
            - hidden_size (int): Dimensionality of the hidden states.
            - num_hidden_layers (int): Number of encoder layers to stack.
            - max_position_embeddings (int): Maximum sequence length for positional embeddings.
            - layer_norm_eps (float): Epsilon value to avoid division by zero in layer normalization.
            - hidden_dropout_prob (float): Dropout probability.
    """

    def __init__(self, config: AutoConfig):
        super().__init__()

        # Embedding layer with token and positional embeddings
        self.embeddings = Embedding(config)

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor after passing through all encoder layers, shape (batch_size, seq_len, hidden_size).
        """
        # Get the embeddings from the input token IDs
        x = self.embeddings(input_ids)

        # Pass the embeddings through each encoder layer
        for layer in self.layers:
            x = layer(x)

        return x
