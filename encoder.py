import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, pos_bias=None):
        """Compute scaled dot-product attention"""
        batch_size, num_heads, seq_len, d_k = Q.size()

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Add positional bias if provided (for relative position encoding)
        if pos_bias is not None:
            scores = scores + pos_bias

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)

        return context, attention_weights

    def forward(self, query, key, value, mask=None, pos_bias=None):
        batch_size, seq_len, d_model = query.size()

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask, pos_bias)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model

        # Create position encodings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).float().unsqueeze(1)

        sinusoid_inp = position * self.inv_freq.unsqueeze(0)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        return sin, cos

    def apply_rotary_pos_emb(self, q, k, sin, cos):
        """Apply rotary positional embeddings to queries and keys"""
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        # Apply rotary embeddings
        q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

        return q_embed, k_embed

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_seq_len=512):
        super(RelativePositionBias, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_seq_len - 1), num_heads))

        # Get pair-wise relative position index for each token
        coords = torch.arange(max_seq_len)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += max_seq_len - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, seq_len):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:seq_len, :seq_len].flatten()
        ].view(seq_len, seq_len, -1)

        # Transpose to get [num_heads, seq_len, seq_len]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        return relative_position_bias.unsqueeze(0)  # Add batch dimension

class RoPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=5000):
        super(RoPEMultiHeadAttention, self).__init__(d_model, num_heads, dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        sin, cos = self.rope(query)
        Q, K = self.rope.apply_rotary_pos_emb(Q, K, sin, cos)

        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights

class RelativeBiasMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=512):
        super(RelativeBiasMultiHeadAttention, self).__init__(d_model, num_heads, dropout)
        self.relative_bias = RelativePositionBias(num_heads, max_seq_len)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Get relative position bias
        pos_bias = self.relative_bias(seq_len)

        # Apply attention with positional bias
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask, pos_bias)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pos_encoding_type="rope"):
        super(EncoderLayer, self).__init__()

        if pos_encoding_type == "rope":
            self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        elif pos_encoding_type == "relative_bias":
            self.self_attn = RelativeBiasMultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, dropout=0.1, pos_encoding_type="rope"):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Only use sinusoidal positional encoding for non-RoPE variants
        if pos_encoding_type != "rope":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Embedding
        x = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding (except for RoPE which is applied in attention)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
