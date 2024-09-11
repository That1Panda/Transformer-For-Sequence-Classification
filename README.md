# Transformer-based Sequence Classification

This project implements a Transformer-based sequence classification model using PyTorch library while focusing on clean code and adding mini-tests on each module.


## Project Structure

```bash
.
├── attention_head.py        # Defines the attention head used in multi-head attention
├── multi_head_attention.py  # Defines the multi-head attention
├── encoder_layer.py         # Implements individual encoder layers
├── embedding.py             # Handles token and positional embeddings
├── encoder.py               # Combines embeddings with multiple encoder layers
├── sequence_classification.py  # Final model combining the encoder and classification head
├── tests/                   # Unit tests for each modules
└── README.md                # Project documentation
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/transformer-sequence-classification.git
   cd transformer-sequence-classification
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

The model uses the Hugging Face `AutoConfig` to manage model settings such as `hidden_size`, `num_labels`, `vocab_size`, and other hyperparameters. You can load a pre-trained configuration or create a custom configuration:

```python
from transformers import AutoConfig

# Load configuration from a pre-trained model
config = AutoConfig.from_pretrained("bert-base-uncased")

# Or create a custom configuration
config = AutoConfig(
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    hidden_dropout_prob=0.1,
    num_labels=2  # For binary classification
)
```

### Model Initialization

After configuring the model, instantiate and use it for sequence classification:

```python
from sequence_classification import SequenceClassification
import torch

# Initialize the model
model = SequenceClassification(config)

# Example input tensor (batch_size=8, seq_len=128)
input_ids = torch.randint(0, config.vocab_size, (8, 128))

# Get the model's output logits
logits = model(input_ids)
print(logits.shape)  # Output: (batch_size, num_labels)
```

### Training & Evaluation

The model is designed to be compatible with PyTorch's training loop. You can fine-tune the model on custom datasets and apply any typical PyTorch optimization techniques.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Forward pass
logits = model(input_ids)
loss = loss_fn(logits, labels)  # Assuming `labels` is the true label tensor

# Backward pass and optimization
loss.backward()
optimizer.step()
```

## Testing

Unit tests are provided to ensure the individual components function as expected. You can run the tests using `pytest`:

```bash
pytest tests/
```

## Notes
This isnt the exact implementation of the Transfomer's paper but rather a simplified version

- Positional encodings are left to be learned by the data
- The encoder layer is implemented with pre-layer normalization

## Acknowledgments

- [Natural Language Processing with Transformers by Lewis Tunstall, Leandro von Werra, Thomas Wolf](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) for the Book that guided me through the project
