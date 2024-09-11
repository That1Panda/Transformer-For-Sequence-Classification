import torch
import torch.nn as nn
from transformers import AutoConfig

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Implements a single encoder layer in the transformer architecture.

    Args:
        - config (transformers.PretrainedConfig): A configuration object from the Hugging Face transformers library.
    """

    def __init__(self, config: AutoConfig):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the encoder layer, of shape (batch_size, seq_len, hidden_size).
        """
        # Apply layer normalization
        x_normalized = self.layer_norm1(x)

        # Apply multi-head attention
        attention_output = self.attention(x_normalized)

        # Add the input to the attention output and apply layer normalization
        x = x + attention_output

        x_normalized = self.layer_norm2(x)

        # Apply feed-forward neural network
        feed_forward_output = self.feed_forward(x_normalized)

        # Add the input to the feed-forward output and return
        return x + feed_forward_output
