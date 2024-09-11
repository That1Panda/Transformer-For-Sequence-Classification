import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """
    Implements a single attention head for multi-head attention.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        head_dim (int): Dimensionality of each attention head.
    """
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention head.

        Args:
            hidden_state (torch.Tensor): Input hidden states of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output of the attention mechanism of shape (batch_size, seq_len, head_dim).
        """
        attn_outputs = self.scaled_dot_product_attention(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )
        return attn_outputs
    
    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Computes scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, head_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, head_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, head_dim).

        Returns:
            torch.Tensor: The weighted sum of values, of shape (batch_size, seq_len, head_dim).
        """
        dim_k = query.size(-1)
        
        # Compute scaled dot-product attention
        scores = torch.bmm(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor((dim_k), dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)
        
        return torch.bmm(weights, value)
