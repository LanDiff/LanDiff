from __future__ import annotations

import logging
import math

import einops
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import LayerNorm

from landiff.modules.pos_emb import apply_rope
from landiff.utils import maybe_autocast

from ...modules.packed_seq import PackedSeqlens
from .inference import get_kvcache_manager

logger = logging.getLogger(__name__)


FLASH_ATTN_IMPORT_ERROR = """
Please install flash_attn V2 to use the Flash Attention module.
""".replace(
    "\n", " "
)

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.warning(FLASH_ATTN_IMPORT_ERROR)


# https://github.com/huggingface/pytorch-image-models/blob/245ad4f41ad23438803c1804763c23a1e447b1ec/timm/layers/drop.py#L137
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlens: PackedSeqlens | None,
    k_seqlens: PackedSeqlens | None,
    causal: bool,
    deterministic: bool = False,
):
    """Multi-head attention.
    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_seqlens: packed seqlens if packing is used. Sum must be equal to q.shape[0].
        k_seqlens: packed seqlens if packing is used. Sum must be equal to k.shape[0].
        causal: use causal mask or not
        logits_cap: use the logits cap or not
        deterministic: Whether to use the deterministic implementation of the backward pass
            when flash atteention is used, which is slightly slower and uses more memory.
            The forward pass is always deterministic.

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
        max_logits: optional scalar tensor
    """

    # FIXME: Do not allow padding in future, record original shape to recover padding from packing
    if q.dim() == 4:
        b, l = q.shape[:2]
        kl = k.shape[1]
    else:
        b = l = None

    # STEP 0: legal check and convert to a unified [tot_seqlens, num_heads, head_dim] format
    # Convert packing to padding
    if q_seqlens is None:
        assert k_seqlens is None, "k_seqlens must be None if q_seqlens is None"
        assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dims if padding"

        # convert to a unified [tot_seqlens, num_heads, head_dim] format
        q = rearrange(q, "b l h d -> (b l) h d")
        k = rearrange(k, "b l h d -> (b l) h d")
        v = rearrange(v, "b l h d -> (b l) h d")
        q_seqlens = PackedSeqlens([l] * b)
        k_seqlens = PackedSeqlens([kl] * b)

    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert (
        q_seqlens is not None and k_seqlens is not None
    ), "q_seqlens and k_seqlens must not be None"
    assert q_seqlens.total_seqlen() == q.shape[0], "q_seqlens must sum to q.shape[0]"
    assert (
        k_seqlens.total_seqlen() == k.shape[0] == v.shape[0]
    ), "k_seqlens must sum to k.shape[0]"

    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"
    attn_func = flash_attn_varlen_func

    # STEP 1: call flash_attn
    attn_out = attn_func(
        q,
        k,
        v,
        q_seqlens.cu_seqlens(q.device),
        k_seqlens.cu_seqlens(k.device),
        q_seqlens.max_seqlen(),
        k_seqlens.max_seqlen(),
        causal=causal,
        deterministic=deterministic,
    )

    # STEP 3: reshape back to original shape if padding
    if b is not None and l is not None:
        attn_out = rearrange(attn_out, "(b l) h d -> b l (h d)", b=b, l=l)
    else:
        attn_out = rearrange(attn_out, "... h d -> ... (h d)")

    return attn_out, None


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.square(x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


class MLP2(nn.Module):

    def __init__(self, dims: list[int], activation, bias=True):
        """
        Args:
            dims: [in_dim, hidden_dim, out_dim]
            bias: whether to use bias in linear layer.
        """
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class LlamaMLP2(nn.Module):
    """https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L307"""

    def __init__(self, dims: list[int], activation=F.silu):
        # dims: [hidden_dim, mlp_dim, hidden_dim]
        super().__init__()
        assert len(dims) == 3
        assert dims[2] == dims[0]
        hidden_dim = dims[0]
        mlp_dim = dims[1]

        self.w1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.activation = activation
        for m in [self.w1, self.w2, self.w3]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        activation=F.gelu,
        *,
        drop_path: float = 0.0,
        attn_bias: bool = False,
        qk_norm: bool = False,
        use_deterministic_attn: bool = False,
    ):
        """
        Args:
            logits_cap: cap the absolute value of logits. From praxis
                https://github.com/google/praxis/blob/8c42aafce511e5f70de74f344b4a92ef3f7c0f1f/praxis/layers/attentions.py#L1127
            attn_bias: whether to use bias in the wqkv and wo linear.
            qk_norm: apply norm to query and key.
            use_deterministic_attn: Whether to use the deterministic implementation of the backward pass
                when flash atteention is used, which is slightly slower and uses more memory.
                The forward pass is always deterministic.
        """
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.num_attention_heads_per_partition = (
            self.num_heads // 1
        )  # Currently we do not support TP!

        self.norm0 = LayerNorm(hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.drop_path = drop_path
        # TODO: use the module from nn/attention.py after supporting causal attn.
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(hidden_dim)
            self.k_norm = LayerNorm(hidden_dim)

        self.use_deterministic_attn = use_deterministic_attn

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        *,
        seqlens: PackedSeqlens | None,
        causal: bool,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            seqlens: the seqlen of x if it's packed.
            rope_freqs_cis: (B, L, D/2) or (L, D/2) or None, the rope frequency cis for each position in the input.
        """
        xqkv = self.wqkv(x)
        if self.qk_norm:
            xq, xk, xv = torch.chunk(xqkv, 3, dim=-1)  # (..., hidden_dim)
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
            xqkv = torch.cat([xq, xk, xv], dim=-1)  # (..., 3*hidden_dim)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size?, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        if rope_freqs_cis is not None:
            xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_out, max_logits = multihead_attention(
            xq,
            xk,
            xv,
            q_seqlens=seqlens,
            k_seqlens=seqlens,
            causal=causal,
            deterministic=self.use_deterministic_attn,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def local_kvcache_inference(
        self,
        x: torch.Tensor,
        current_kvcache_manager,
        prefix_paddings=None,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
          x: (batch, seqlen, dim)
          current_kvcache_manager: current kv_cache manager
          prefix_paddings: (batch, prefix_length), default as None. If not None, the prefix_paddings is correspondent to the prefix sequence (the condition), which means that during the inference or sampling process, the prefix_paddings remains UNChanged. In this prefix_paddings, 1 means valid tokens, 0 means padding tokens.
        """
        xqkv = self.wqkv(x)
        if self.qk_norm:
            xq, xk, xv = torch.chunk(xqkv, 3, dim=-1)  # (..., hidden_dim)
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
            xqkv = torch.cat([xq, xk, xv], dim=-1)  # (..., 3*hidden_dim)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, m, 3, nheads, headdim), m = 1 (in generation) or m = seqlen (in prefill)
        xqkv = xqkv.view(*qkv_shape)
        _, num_tokens, _, _, head_dim = xqkv.shape
        query, key, value = torch.unbind(
            xqkv, dim=2
        )  # (batch_size, m, nheads, headdim), m = 1 (in generation) or m = seqlen (in prefill)
        if rope_freqs_cis is not None:
            query, key = apply_rope(query, key, rope_freqs_cis)
        kv_cache = current_kvcache_manager.get_kvcache(self.__scope__)
        if kv_cache is not None:
            assert key.shape[1] == 1
            past_key, past_value = kv_cache
            key = torch.cat(
                [past_key, key], dim=1
            )  # (batch_size, current_seqlen, nheads, headdim)
            value = torch.cat([past_value, value], dim=1)
        kv_cache = torch.stack(
            [key, value]
        )  # (2, batch_size, current_seqlen, nheads, headdim)
        current_kvcache_manager.set_kvcache(self.__scope__, kv_cache)

        scores = torch.einsum("nqhd,nkhd->nhqk", [query, key]) / (
            head_dim**0.5
        )  # (batch_size, nheads, 1, seqlen)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        max_neg_value = -torch.finfo(
            scores.dtype
        ).max  # do not use -inf because when any row of the mask is all True, the value after softmax will be 0
        if (
            scores.shape[-2] > 1
        ):  # in this case, there are more than one conditional token, we need to apply causal mask when we train the model causally
            mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)

        if prefix_paddings is not None:
            prefix_paddings = F.pad(
                prefix_paddings,
                (0, scores.shape[-1] - prefix_paddings.shape[-1]),
                value=True,
            )  # pad to the right to match the current seqlen
            prefix_paddings = einops.repeat(
                prefix_paddings, "b n -> b () m n", m=num_tokens
            )
            mask = torch.max(mask, ~prefix_paddings)
        scores.masked_fill_(mask, max_neg_value)  # mask the pad tokens

        attention = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn_out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).flatten(
            start_dim=2
        )  # (batch_size, 1, nheads*headdim)
        attn_out = self.wo(attn_out)  # (batch_size, 1, nheads*headdim)
        return attn_out

    def forward(
        self,
        input: torch.Tensor,
        *,
        causal: bool = False,
        seqlens: PackedSeqlens | None = None,
        prefix_paddings: torch.Tensor = None,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            input/output shape: (batch if not packing, seqlen, dim)
            causal: if False, full attention is used.
            prefix_paddings: (batch, prefix_length), default as None. If not None, the prefix_paddings is correspondent to the prefix sequence (the condition), which means that during the inference or sampling process, the prefix_paddings remains UNChanged. In this prefix_paddings, 1 means valid tokens, 0 means padding tokens.
        """
        if seqlens is None:
            assert (
                input.dim() == 3
            ), f"Expected (batch_size, seq_length, hidden_dim), got {input.shape}"
        else:
            assert (
                input.dim() == 2
            ), f"Expected (seq_length, hidden_dim), got {input.shape}"
        x = self.norm0(input)
        with maybe_autocast(x, torch.bfloat16):
            if (
                not self.training
                and (current_kvcache_manager := get_kvcache_manager())
                and causal
            ):
                assert seqlens is None
                attn_out = self.local_kvcache_inference(
                    x,
                    current_kvcache_manager,
                    prefix_paddings,
                    rope_freqs_cis=rope_freqs_cis,
                )
            else:
                assert prefix_paddings is None, prefix_paddings
                attn_out = self.attention_qkvpacked(
                    x, causal=causal, seqlens=seqlens, rope_freqs_cis=rope_freqs_cis
                )
            input = input + drop_path(
                attn_out, drop_prob=self.drop_path, training=self.training
            )
            x = self.mlp(self.norm1(input))
            input = input + drop_path(
                x, drop_prob=self.drop_path, training=self.training
            )
        return input


class LlamaTransformerBlock(TransformerBlock):

    def __init__(
        self, num_heads: int, hidden_dim: int, mlp_dim: int, activation=F.silu, **kwargs
    ):
        super().__init__(num_heads, hidden_dim, mlp_dim, activation, **kwargs)
        self.norm0 = RMSNorm(
            hidden_dim
        )  # params_dtype: fp32, the default params_dtype is bfloat16
        self.norm1 = RMSNorm(hidden_dim)
        self.mlp = LlamaMLP2([hidden_dim, mlp_dim, hidden_dim], activation)

    # TODO: apply ROPE in every layer?
