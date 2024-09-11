import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import pytest
from attention_head import AttentionHead  


@pytest.mark.parametrize("batch_size, seq_len, embed_dim, head_dim", [
    (2, 10, 64, 16),  # Test case 1
    (4, 20, 128, 32),  # Test case 2
    (1, 5, 256, 64),  # Test case 3
])
def test_attention_head_output_size(batch_size, seq_len, embed_dim, head_dim):
    # Initialize the AttentionHead
    attention_head = AttentionHead(embed_dim=embed_dim, head_dim=head_dim)
    
    # Create a dummy input tensor with the correct shape
    hidden_state = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = attention_head(hidden_state)
    
    # Assert that the output shape is as expected
    assert output.shape == (batch_size, seq_len, head_dim), \
        f"Expected output shape {(batch_size, seq_len, head_dim)} but got {output.shape}"

