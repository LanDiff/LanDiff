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

from ...modules.packed_seq import PackedSeqlens
from ..modules.tokens import SequenceBuilder

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
        # parti: "all layer norms and model output are kept as float32"
        # This is not supported under fsdp1 - the entire model must share a mixed precision policy.
        # But TritonLN already uses float32 for reduction, so it should be close.
        # x = x.float()
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
        prefix_paddings: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
          x: In prefill, shape [batch_size, tokens_num, hidden_dim].
            In generation, last token's feature of shape [batch_size, 1, hidden_dim]
          prefix_paddings: (batch, prefix len), bool tensor. If not None,
            the padding mask of the prefix sequence. This means that during sampling,
            it remains unchanged. This is used in the attention module.
            1 means valid tokens, 0 means padding tokens.
        Returns:
            Last token's prediction, of shape [batch_size, 1, vocab_size].

        This function should be called under a KVCache manager.
        """
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
                    prefix_paddings=prefix_paddings,
                    rope_freqs_cis=freqs_cis,
                )
        assert (
            x.dtype == self.fwd_dtype
        ), f"feature dtype: {x.dtype}, fwd_dtype: {self.fwd_dtype}"
        # parti: "all layer norms and model output are kept as float32"
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
        all_seqs = self.tokenize(x)
        seqlens = [len(x) for x in all_seqs]

        all_seqs = all_seqs[0].cat(all_seqs, keep_name=False)
        tokens = all_seqs.concat_tokens()  # [seqlen]
        embeddings = all_seqs.concat_features(None)  # [seqlen, dim]
        loss_mask = all_seqs.concat_loss_mask()  # [seqlen]

        # Shift sequence by 1 to perform next-token prediction.
        seqlens[-1] -= 1
        packed_seqlens = PackedSeqlens(seqlens)
        # For packing, the first token of every seq must have loss_mask=False.
        loss_mask[packed_seqlens.cu_seqlens(loss_mask.device)[1:-1]] = False
        labels, loss_mask = tokens[1:], loss_mask[1:]

        if not self.training:
            return (
                self.transformer(embeddings[:-1], seqlens=packed_seqlens),
                labels,
                loss_mask,
            )  # [tot_seqlen-1, vocab_size]

        if self.use_chunked_cross_entropy:
            pass
        else:
            logits = self.transformer(embeddings[:-1], seqlens=packed_seqlens)
            losses = self._losses(logits, labels, loss_mask)

        # TODO: implement visualization
        return losses

    def _losses(self, logits, labels, loss_mask):
        logits, labels = logits[loss_mask], labels[loss_mask]
        loss = F.cross_entropy(logits, labels)
        losses = {"cross_entropy": loss}
        return losses

    def forward_padding(self, x):
        all_seqs = self.tokenize(x)
        all_seqs = SequenceBuilder.pad_to_longest(
            all_seqs, padding_token=self.vocab.PAD, padding_side="right"
        )
        tokens, embeddings, loss_mask = SequenceBuilder.batch(all_seqs)
        # [N, seqlen], [N, seqlen, hidden_dim], [N, seqlen]

        loss_start_idx = min(seq.get_range("visual")[0] for seq in all_seqs)
        # Shift sequence by 1 to perform next-token prediction.
        logits = self.transformer(embeddings[:, :-1])  # [N, seqlen-1, vocab_size]
        logits = logits[:, loss_start_idx - 1 :]
        labels, loss_mask = tokens[:, loss_start_idx:], loss_mask[:, loss_start_idx:]

        if self.training:
            losses = self._losses(logits, labels, loss_mask)

            return losses
        return logits, all_seqs

    def decode_logits_argmax(self, logits, loss_mask):
        """Decode the argmax prediction from logits.

        logits: (N, seqlen-1, vocab_size). TODO: support packing.
        """
        # Assert that every sample has the same number of loss-enabled tokens.
        # This is required so we can safely do reshape after the indexing.
        assert (loss_mask.sum(dim=1) == loss_mask[0].sum()).all()
        codes = torch.argmax(logits, dim=-1)[loss_mask].reshape(logits.shape[0], -1)
        # Clamp because old configs have N+1 classes.
        codes.clamp_(min=0, max=self.tokenizer.vocab_size() - 1)
        # Assume that only visual tokens are loss-enabled.
        reconstructed = self.tokenizer.decode_codes(codes)
        return reconstructed

    def forward_reconstruction(self, x):
        """Run teacher-forcing inference with argmax decoding to reconstruct the input."""
        logits, all_seqs = self.forward_padding(x)
        _, _, loss_mask = SequenceBuilder.batch(all_seqs)
        loss_start_idx = min(seq.get_range("visual")[0] for seq in all_seqs)
        loss_mask = loss_mask[:, loss_start_idx:]
        reconstructed = self.decode_logits_argmax(logits, loss_mask)
        return reconstructed

    def forward(self, x):
        """
        Args:
          x: a dict with optionally the following keys:
            "image": NCHW tensor or "video": NCDHW tensor
            "class_id": shape [N] integers, or "caption": list[str]
        """
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
