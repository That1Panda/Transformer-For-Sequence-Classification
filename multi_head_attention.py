import torch
import torch.nn as nn
from attention_head import AttentionHead

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention by utilizing several attention heads and combining their outputs.

    Args:
        config (object): Configuration object containing attributes `hidden_size` (int) and `num_attention_heads` (int).
    """
    def __init__(self, config: object):
        super().__init__()
        
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        
        # Linear layer to combine the output of all heads
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            hidden_state (torch.Tensor): Input hidden states of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output of the multi-head attention of shape (batch_size, seq_len, embed_dim).
        """
        # Apply attention for each head and concatenate the results
        x = torch.cat([head(hidden_state) for head in self.heads], dim=-1)
        
        # Apply a linear transformation on the concatenated results
        x = self.output_linear(x)
        
        return x
