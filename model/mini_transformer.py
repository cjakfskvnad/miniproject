import torch
import torch.nn as nn
import math


class MiniTransformer(nn.Module):
    """
    A minimal Decoder-only Transformer model (GPT-style) with 3-token vocabulary: 0, 1, sep
    """
    def __init__(self, vocab_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=128):
        super(MiniTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer decoder (decoder-only architecture)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len):
        """
        Generate causal mask for autoregressive generation
        Returns upper triangular matrix filled with -inf
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len)
        Returns:
            output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Generate causal mask for autoregressive decoding
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        # Transformer decoding (self-attention with causal mask)
        # In decoder-only model, we use the same input as both tgt and memory
        x = self.transformer_decoder(x, x, tgt_mask=causal_mask)

        # Output projection
        output = self.fc_out(x)

        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


if __name__ == "__main__":
    # Create model with random parameters
    model = MiniTransformer(
        vocab_size=3,      # 0, 1, sep
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_seq_len=128
    )

    # Save the model
    torch.save(model.state_dict(), 'model/mini_transformer_weights.pth')
    torch.save(model, 'model/mini_transformer_full.pth')

    print("Mini Decoder-only Transformer model created and saved!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nModel architecture:")
    print(model)

    # Test with sample input
    batch_size = 2
    seq_len = 10
    sample_input = torch.randint(0, 3, (batch_size, seq_len))

    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Sample input:\n{sample_input}")

    with torch.no_grad():
        output = model(sample_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output logits (first sequence, first 3 positions):\n{output[0, :3, :]}")

    # Show that causal mask is working
    print(f"\nCausal mask (5x5 example):")
    print(model.generate_causal_mask(5))
