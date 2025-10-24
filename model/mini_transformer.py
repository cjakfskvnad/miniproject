import torch
import torch.nn as nn
import math
from flash_attn import flash_attn_func


class FlashAttentionBlock(nn.Module):
    """
    Transformer block using Flash Attention for efficient self-attention
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(FlashAttentionBlock, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Q, K, V projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            causal: whether to use causal masking for autoregressive generation
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention expects (batch, seq_len, nhead, head_dim)
        q = q.transpose(1, 2)  # (batch, seq_len, nhead, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention only supports fp16 and bf16
        # Store original dtype and convert
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            # Use bf16 if available (better for training), otherwise fp16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16

            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        else:
            target_dtype = orig_dtype

        # Apply Flash Attention
        attn_output = flash_attn_func(q, k, v, causal=causal)

        # Convert back to original dtype if needed
        if target_dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Add residual
        x = residual + attn_output

        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.ff(x)
        ff_output = self.dropout(ff_output)
        x = residual + ff_output

        return x


class MiniTransformer(nn.Module):
    """
    A minimal Decoder-only Transformer model (GPT-style) with 3-token vocabulary: 0, 1, sep
    """
    def __init__(self, vocab_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=128, dropout=0.1):
        super(MiniTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks with Flash Attention
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

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

        # Apply Flash Attention blocks (causal masking is handled inside)
        for block in self.blocks:
            x = block(x, causal=True)

        # Final layer norm
        x = self.ln_f(x)

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
