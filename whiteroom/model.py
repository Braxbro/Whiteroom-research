"""
Small encoder-decoder transformer for the whiteroom task.

Encoder reads: [A tokens] [BIND port_idx_A port_idx_B] [B tokens]
Decoder generates: [port_tokens...] [op_tokens...] [flag_tokens...] [END]

Two heads:
  - Sequence head: linear(d_model → vocab_size) on decoder output
  - is_valid head: linear(d_model → 1) on mean-pooled encoder output
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .generator import VOCAB_SIZE
from .vocab import Token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WhiteroomTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        causal_encoder: bool = False,
        block_diag_encoder_mask: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.causal_encoder = causal_encoder
        self.block_diag_encoder_mask = block_diag_encoder_mask

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=Token.PAD)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=Token.PAD)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.seq_head = nn.Linear(d_model, vocab_size)
        self.valid_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        src: (batch, src_len) token ids
        returns memory: (batch, src_len, d_model)
        """
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        src_mask = None

        if self.causal_encoder:
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                src.size(1), device=src.device)
        elif self.block_diag_encoder_mask:
            # Block-diagonal: A and B isolated, BIND bridges them
            # PyTorch Transformer: True = mask out (don't attend), False = attend
            # Find BIND token (Token.BIND=3) in the first sequence
            seq_len = src.size(1)
            bind_idx = None
            for i in range(seq_len):
                if src[0, i].item() == Token.BIND:
                    bind_idx = i
                    break

            if bind_idx is not None:
                # Start with all positions masked out
                src_mask = torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool)

                # A tokens (0:bind_idx) can attend to: A tokens (0:bind_idx) + BIND (bind_idx)
                src_mask[0:bind_idx, 0:bind_idx+1] = False

                # BIND (bind_idx) can attend to everything
                src_mask[bind_idx, :] = False

                # B tokens (bind_idx+1:seq_len) can attend to: BIND (bind_idx) + B tokens (bind_idx+1:seq_len)
                src_mask[bind_idx+1:, bind_idx:] = False

        return self.transformer.encoder(x, mask=src_mask,
                                        src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        tgt: (batch, tgt_len) token ids
        memory: (batch, src_len, d_model)
        returns: (batch, tgt_len, d_model)
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        src: (batch, src_len)  — input token ids
        tgt: (batch, tgt_len)  — target token ids (teacher-forced; BOS-prepended)

        Returns:
            seq_logits:   (batch, tgt_len, vocab_size)
            valid_logits: (batch, 1)
        """
        memory = self.encode(src, src_key_padding_mask)

        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)

        dec_out = self.decode(tgt, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

        seq_logits = self.seq_head(dec_out)  # (batch, tgt_len, vocab_size)

        # is_valid: mean-pool encoder output over non-padding positions
        if src_key_padding_mask is not None:
            # mask is True where padding — invert for mean pooling
            not_pad = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (memory * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp(min=1)
        else:
            pooled = memory.mean(dim=1)
        valid_logits = self.valid_head(pooled)  # (batch, 1)

        return seq_logits, valid_logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        max_len: int = 32,
    ) -> Tensor:
        """
        Greedy autoregressive decode.
        Returns predicted token ids: (batch, decoded_len)
        """
        memory = self.encode(src, src_key_padding_mask)
        batch = src.size(0)
        device = src.device

        # Start with BOS = Token.COMPOUND
        ys = torch.full((batch, 1), Token.COMPOUND, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            dec_out = self.decode(ys, memory, tgt_mask=tgt_mask,
                                  memory_key_padding_mask=src_key_padding_mask)
            logits = self.seq_head(dec_out[:, -1, :])  # (batch, vocab_size)
            next_tok = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == Token.END)
            if finished.all():
                break

        return ys[:, 1:]  # strip BOS


class WhiteroomTransformer3Stage(WhiteroomTransformer):
    """
    3-stage transformer: Encoder -> Adaptation -> Decoder

    Inherits from WhiteroomTransformer to be compatible with existing eval scripts.
    Overrides encode/decode/forward to implement 3-stage architecture with explicit
    block-diagonal encoder and adaptation layer.

    Architecture:
    - Stage 1 (Encoder): Block-diagonal bidirectional attention (isolates A and B)
    - Stage 2 (Adaptation): MLP layer to bridge encoder→decoder representation spaces
    - Stage 3 (Decoder): 3 layers of cross-attention, fully free attention
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        # Accept parent's parameters (won't be used)
        causal_encoder: bool = False,
        block_diag_encoder_mask: bool = False,
    ):
        # Don't call super().__init__() — build 3-stage architecture instead
        nn.Module.__init__(self)
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=Token.PAD)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=Token.PAD)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stage 1: Encoder with block-diagonal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Stage 2: Adaptation layer (MLP to give it actual capacity)
        self.adaptation = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        # Stage 3: Decoder with free attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.seq_head = nn.Linear(d_model, vocab_size)
        self.valid_head = nn.Linear(d_model, 1)

        # Create mock transformer attribute for eval script compatibility
        self.transformer = type('obj', (object,), {
            'encoder': self.encoder,
            'decoder': self.decoder,
        })()

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Stage 1: Encode with block-diagonal attention (A and B isolated by BIND).

        src: (batch, src_len) token ids
        returns memory: (batch, src_len, d_model)
        """
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))

        # Block-diagonal mask: A and B isolated, BIND bridges them
        seq_len = src.size(1)
        bind_idx = None
        for i in range(seq_len):
            if src[0, i].item() == Token.BIND:
                bind_idx = i
                break

        src_mask = None
        if bind_idx is not None:
            # Start with all positions masked out (True = mask)
            src_mask = torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool)

            # A tokens (0:bind_idx) can attend to: A tokens (0:bind_idx) + BIND (bind_idx)
            src_mask[0:bind_idx, 0:bind_idx+1] = False

            # BIND (bind_idx) can attend to everything
            src_mask[bind_idx, :] = False

            # B tokens (bind_idx+1:seq_len) can attend to: BIND (bind_idx) + B tokens (bind_idx+1:seq_len)
            src_mask[bind_idx+1:, bind_idx:] = False

        return self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def adapt(self, memory: Tensor) -> Tensor:
        """
        Stage 2: Adaptation layer bridges encoder→decoder representation space.

        memory: (batch, src_len, d_model)
        returns: (batch, src_len, d_model)
        """
        return self.adaptation(memory)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Stage 3: Decode with cross-attention to adapted memory.
        Decoder has causal self-attention but free cross-attention to memory.

        tgt: (batch, tgt_len) token ids
        memory: (batch, src_len, d_model) from adapt()
        tgt_mask: optional causal mask (defaults to generated if None)
        returns: (batch, tgt_len, d_model)
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        tgt_len = tgt.size(1)

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)

        return self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full 3-stage forward pass: Encode → Adapt → Decode

        src: (batch, src_len)
        tgt: (batch, tgt_len)

        Returns:
            seq_logits:   (batch, tgt_len, vocab_size)
            valid_logits: (batch, 1)
        """
        # Stage 1: Encode with block-diagonal attention
        memory = self.encode(src, src_key_padding_mask)

        # Stage 2: Adapt to bridge representation spaces
        memory = self.adapt(memory)

        # Stage 3: Decode
        dec_out = self.decode(tgt, memory,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)

        seq_logits = self.seq_head(dec_out)  # (batch, tgt_len, vocab_size)

        # is_valid: mean-pool encoder output over non-padding positions
        if src_key_padding_mask is not None:
            not_pad = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (memory * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp(min=1)
        else:
            pooled = memory.mean(dim=1)
        valid_logits = self.valid_head(pooled)  # (batch, 1)

        return seq_logits, valid_logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        max_len: int = 32,
    ) -> Tensor:
        """Greedy autoregressive decode."""
        memory = self.encode(src, src_key_padding_mask)
        memory = self.adapt(memory)

        batch = src.size(0)
        device = src.device

        ys = torch.full((batch, 1), Token.COMPOUND, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            dec_out = self.decode(ys, memory,
                                 memory_key_padding_mask=src_key_padding_mask)
            logits = self.seq_head(dec_out[:, -1, :])  # (batch, vocab_size)
            next_tok = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == Token.END)
            if finished.all():
                break

        return ys[:, 1:]  # strip BOS
