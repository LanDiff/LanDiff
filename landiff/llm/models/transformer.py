from __future__ import annotations

import logging
from ast import List
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from landiff.modules.pos_emb import Rope1DPosEmb
from landiff.utils import maybe_autocast

logger = logging.getLogger(__name__)


class GPT(nn.Module):

    def __init__(
        self,
        visual_vocab_size: int,
        hidden_dim: int,
        blocks: List[nn.Module],
        causal: bool = True,
        fwd_dtype: torch.dtype = torch.float32,
        rope: Rope1DPosEmb | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visual_vocab_size = visual_vocab_size
        self.causal = causal
        self.fwd_dtype = fwd_dtype

        # transformer blocks
        self.blocks = blocks
        self.blocks = nn.Sequential(*self.blocks)
        self.rope = rope

        # head
        self.layer_norm = LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, visual_vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, seqlens: list[int] | None = None, apply_head=True):
        """
        Args:
          x: tensor of shape [batch_size, seqlen, hidden_dim],
            or [total_seqlen, hidden_dim] if packing.
          seqlens: sequence length if packing.
          apply_head: return logits if true, or return norm output & head
        """
        x = x.to(dtype=self.fwd_dtype)
        if self.rope is not None:
            assert isinstance(
                self.rope, Rope1DPosEmb
            ), "Only 1D pos emb is supported for now"
            assert seqlens is not None, "seqlens must be provided for 1D pos emb"
            freqs_cis = self.rope.get_freqs_cis_by_seqlens(seqlens).to(x.device)
        else:
            freqs_cis = None
        with maybe_autocast(x, torch.bfloat16):
            for block in self.blocks:
                x = block(
                    x, seqlens=seqlens, causal=self.causal, rope_freqs_cis=freqs_cis
                )
        assert (
            x.dtype == self.fwd_dtype
        ), f"feature dtype: {x.dtype}, fwd_dtype: {self.fwd_dtype}"
        x = self.layer_norm(x)
        if not apply_head:
            return x
        logits = self.hidden2logits(x)
        return logits

    def hidden2logits(self, x):
        with maybe_autocast(x, self.fwd_dtype):
            logits = self.head(x)
        return logits

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
    ):
        x = x.to(self.fwd_dtype)
        if self.rope is not None:
            assert isinstance(
                self.rope, Rope1DPosEmb
            ), "Only 1D pos emb is supported for now"
            assert freqs_cis is not None, "freqs_cis must be provided for 1D pos emb"
        else:
            freqs_cis = None
        with maybe_autocast(x, torch.bfloat16):
            for block in self.blocks:
                x = block(
                    x,
                    causal=self.causal,
                    rope_freqs_cis=freqs_cis,
                )
        assert (
            x.dtype == self.fwd_dtype
        ), f"feature dtype: {x.dtype}, fwd_dtype: {self.fwd_dtype}"
        x = x.float()
        x = self.layer_norm(x)
        x = x[:, -1].contiguous()
        logits = self.head(x)
        return logits


class CondTransformerBase(nn.Module):

    def __init__(self, use_chunked_cross_entropy=False):
        super().__init__()
        self.use_chunked_cross_entropy = use_chunked_cross_entropy

    def tokenize(self, x):
        raise NotImplementedError

    def forward_packing(self, x):
        raise NotImplementedError

    def _losses(self, logits, labels, loss_mask):
        logits, labels = logits[loss_mask], labels[loss_mask]
        loss = F.cross_entropy(logits, labels)
        losses = {"cross_entropy": loss}
        return losses

    def forward(self, x):
        return self.forward_packing(x)

    @torch.no_grad()
    def sample(
        self,
        inputs: dict[str, Any],
        top_k: float | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ):
        raise NotImplementedError
