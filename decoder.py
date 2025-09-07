import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import (MultiHeadAttention, RoPEMultiHeadAttention,
                    RelativeBiasMultiHeadAttention, PositionwiseFeedForward,
                    SinusoidalPositionalEncoding, TransformerEncoder)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pos_encoding_type="rope"):
        super(DecoderLayer, self).__init__()

        # Self-attention in decoder
        if pos_encoding_type == "rope":
            self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        elif pos_encoding_type == "relative_bias":
            self.self_attn = RelativeBiasMultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = RelativeBiasMultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with look-ahead mask
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, dropout=0.1, pos_encoding_type="rope"):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Only use sinusoidal positional encoding for non-RoPE variants
        if pos_encoding_type != "rope":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding (except for RoPE which is applied in attention)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Output projection
        output = self.output_projection(x)

        return output

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_len=512, dropout=0.1, pos_encoding_type="rope"):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type
        )

        self.src_pad_idx = 0
        self.tgt_pad_idx = 0

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return decoder_output

class BeamSearchDecoder:
    def __init__(self, model, beam_size=4, max_length=100, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def decode(self, src, src_mask=None):
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.model.encoder(src, src_mask)

        # Initialize beams
        beams = torch.full((batch_size, self.beam_size, 1), self.sos_idx,
                          dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        beam_scores[:, 1:] = -float('inf')  # Only first beam is active initially

        finished_beams = []

        for step in range(self.max_length):
            # Expand beams for current step
            current_beams = beams.view(batch_size * self.beam_size, -1)

            # Expand encoder output for beam search
            expanded_encoder_output = encoder_output.unsqueeze(1).expand(
                batch_size, self.beam_size, -1, -1
            ).contiguous().view(batch_size * self.beam_size, -1, encoder_output.size(-1))

            # Expand source mask
            if src_mask is not None:
                expanded_src_mask = src_mask.unsqueeze(1).expand(
                    batch_size, self.beam_size, -1, -1
                ).contiguous().view(batch_size * self.beam_size, -1, -1)
            else:
                expanded_src_mask = None

            # Get decoder output
            tgt_mask = self.model.make_tgt_mask(current_beams)
            decoder_output = self.model.decoder(current_beams, expanded_encoder_output,
                                              expanded_src_mask, tgt_mask)

            # Get probabilities for next token
            next_token_logits = decoder_output[:, -1, :]  # Last position
            next_token_probs = F.log_softmax(next_token_logits, dim=-1)

            # Reshape back to batch x beam
            next_token_probs = next_token_probs.view(batch_size, self.beam_size, -1)

            # Calculate scores
            vocab_size = next_token_probs.size(-1)
            beam_scores_expanded = beam_scores.unsqueeze(-1).expand(-1, -1, vocab_size)
            scores = beam_scores_expanded + next_token_probs

            # Flatten and get top-k
            scores_flat = scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(scores_flat, self.beam_size, dim=-1)

            # Convert back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beams
            new_beams = []
            new_scores = []

            for batch_idx in range(batch_size):
                batch_beams = []
                batch_scores = []

                for beam_idx in range(self.beam_size):
                    old_beam_idx = beam_indices[batch_idx, beam_idx]
                    token_idx = token_indices[batch_idx, beam_idx]

                    # Get the sequence from old beam
                    old_sequence = beams[batch_idx, old_beam_idx]
                    new_sequence = torch.cat([old_sequence, token_idx.unsqueeze(0)])

                    batch_beams.append(new_sequence)
                    batch_scores.append(top_scores[batch_idx, beam_idx])

                new_beams.append(torch.stack([seq.pad(self.max_length, value=self.pad_idx)[:step+2]
                                            for seq in batch_beams]))
                new_scores.append(torch.stack(batch_scores))

            beams = torch.stack(new_beams)
            beam_scores = torch.stack(new_scores)

            # Check for finished sequences
            finished_mask = (token_indices == self.eos_idx)
            if finished_mask.any():
                for batch_idx in range(batch_size):
                    for beam_idx in range(self.beam_size):
                        if finished_mask[batch_idx, beam_idx]:
                            finished_beams.append({
                                'tokens': beams[batch_idx, beam_idx],
                                'score': beam_scores[batch_idx, beam_idx]
                            })

        # Return best beam for each batch
        best_sequences = []
        for batch_idx in range(batch_size):
            best_beam_idx = torch.argmax(beam_scores[batch_idx])
            best_sequences.append(beams[batch_idx, best_beam_idx])

        return torch.stack(best_sequences)

class GreedyDecoder:
    def __init__(self, model, max_length=100, sos_idx=1, eos_idx=2):
        self.model = model
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def decode(self, src, src_mask=None):
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.model.encoder(src, src_mask)

        # Initialize decoder input with SOS token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, device=device)

        for _ in range(self.max_length):
            tgt_mask = self.model.make_tgt_mask(decoder_input)
            decoder_output = self.model.decoder(decoder_input, encoder_output, src_mask, tgt_mask)

            # Get next token
            next_token_logits = decoder_output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Check if all sequences have EOS token
            if (next_token == self.eos_idx).all():
                break

        return decoder_input

class TopKDecoder:
    def __init__(self, model, k=40, max_length=100, sos_idx=1, eos_idx=2, temperature=1.0):
        self.model = model
        self.k = k
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.temperature = temperature

    def decode(self, src, src_mask=None):
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.model.encoder(src, src_mask)

        # Initialize decoder input with SOS token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, device=device)

        for _ in range(self.max_length):
            tgt_mask = self.model.make_tgt_mask(decoder_input)
            decoder_output = self.model.decoder(decoder_input, encoder_output, src_mask, tgt_mask)

            # Get next token logits
            next_token_logits = decoder_output[:, -1, :] / self.temperature

            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, self.k, dim=-1)

            # Apply softmax to top-k logits
            top_k_probs = F.softmax(top_k_logits, dim=-1)

            # Sample from top-k distribution
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices.gather(-1, sampled_indices)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Check if all sequences have EOS token
            if (next_token.squeeze(-1) == self.eos_idx).all():
                break

        return decoder_input
