"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""

import functools
import logging
from collections import OrderedDict, namedtuple
from enum import IntEnum

import einops
import numpy as np
import torch

# modified from titok to support semantic input
import torch.nn as nn
from torch.nn import LayerNorm

from ...modules.pos_emb import Rope3DPosEmb, apply_rope

try:
    import torch.nn.attention.flex_attention as flex_attention_mod

    flex_attention = torch.compile(
        flex_attention_mod.flex_attention
    )  # flex_attention must be compiled
    # flex_attention = flex_attention_mod.flex_attention  # flex_attention must be compiled
    from . import flex_attention_mask as flex_attention_mask_mod
except ImportError:
    flex_attention_mod = None
    flex_attention_mask_mod = None
    flex_attention = None

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None

logger = logging.getLogger(__name__)
Titok_Cfg = namedtuple("Config", ["width", "num_layers", "num_heads"])
TITOK_ARCH_DICT = {
    "small": Titok_Cfg(512, 8, 8),
    "base": Titok_Cfg(768, 12, 12),
    "large": Titok_Cfg(1024, 24, 16),
}


class AttentionImp(IntEnum):
    """Different separator style."""

    FLASH_ATTENTION = 1
    TORCH = 2
    FLEX_ATTENTION = 3


class PositionalEmbedingType(IntEnum):
    """Different positional embedding type."""

    LEARNABLE = 1
    # 3d rope
    ROPE_3D = 2


class AttentionMaskType(IntEnum):
    FULL = 1
    CAUSAL = 2
    PREFIX_LM_CAUSAL = 3
    VIDEO_ENCODER_MASK = 4
    VIDEO_DECODER_MASK = 5


def gen_attention_mask_img(attention_mask: torch.Tensor):
    if isinstance(attention_mask, torch.Tensor):
        assert (
            attention_mask.ndim == 2
        ), f"Expected 2D tensor, got {attention_mask.shape}"
    else:
        raise NotImplementedError(
            f"Attention mask type {type(attention_mask)} not implemented."
        )
    img_array = flex_attention_mask_mod.vis_func(attention_mask)
    img_array = torch.from_numpy(img_array) / 255.0
    img_array = einops.rearrange(img_array, "h w c -> 1 c h w")
    return img_array


class MultiheadAttention(nn.Module):
    """A multihead-attention module for self and cross-attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        bias: bool = False,
        qk_norm: bool = False,
        out_bias: bool | None = None,
        causal: bool = False,
        use_deterministic_attn: bool = False,
        attention_imp: AttentionImp = AttentionImp.FLASH_ATTENTION,
    ):
        super().__init__()
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(hidden_dim)
            self.k_norm = LayerNorm(hidden_dim)
        if out_bias is None:
            out_bias = bias
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=out_bias)
        self.num_heads = num_heads
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        self.dim_per_head = hidden_dim // num_heads
        self.use_deterministic_attn = use_deterministic_attn
        self.causal = causal
        self.attention_imp = attention_imp

        if self.attention_imp == AttentionImp.FLEX_ATTENTION:
            # 要求torch的版本大于等于2.5
            assert torch.__version__ >= "2.5.0", "Flex Attention requires torch>=2.5.0"
            assert flex_attention_mod is not None, "Flex Attention is not available."
        elif self.attention_imp == AttentionImp.FLASH_ATTENTION:
            assert flash_attn_func is not None, "Flash Attention is not available."

    @torch.profiler.record_function("MultiheadAttention.forward")
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        *,
        rope_freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if self.causal and self.attention_imp != AttentionImp.FLEX_ATTENTION:
            assert (
                attention_mask is None
            ), "Causal attention does not support attention_mask."
        if attention_mask is not None:
            assert (
                self.attention_imp != AttentionImp.FLASH_ATTENTION
            ), "Flex Attention does not support attention_mask."
        if kv is not None:
            assert kv.ndim == 3, f"Padding expects 3D input, got {kv.shape}."
        if kv is None:
            x = q
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
        else:
            q = self.wq(q)
            k, v = self.wk(kv), self.wv(kv)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = einops.rearrange(q, "... l (n d) -> ... l n d", n=self.num_heads)
        k = einops.rearrange(k, "... l (n d) -> ... l n d", n=self.num_heads)
        v = einops.rearrange(v, "... l (n d) -> ... l n d", n=self.num_heads)
        if rope_freqs_cis is not None:
            assert (
                rope_freqs_cis.shape[-1] == self.dim_per_head // 2
            ), f"Expected rope_freqs_cis.shape[-1] == {self.dim_per_head // 2}, got {rope_freqs_cis.shape}"
            q, k = apply_rope(q, k, rope_freqs_cis)
        if self.attention_imp == AttentionImp.FLASH_ATTENTION:
            attn_out = flash_attn_func(
                q, k, v, causal=self.causal, deterministic=self.use_deterministic_attn
            )
            attn_out = einops.rearrange(attn_out, "... l n d -> ... l (n d)")
        elif self.attention_imp == AttentionImp.TORCH:
            q = einops.rearrange(q, "... l n d -> ... n l d")
            k = einops.rearrange(k, "... l n d -> ... n l d")
            v = einops.rearrange(v, "... l n d -> ... n l d")
            if attention_mask is not None:
                attn_out = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask
                )
            else:
                attn_out = nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=self.causal
                )
            attn_out = einops.rearrange(attn_out, "... n l d -> ... l (n d)")
        elif self.attention_imp == AttentionImp.FLEX_ATTENTION:
            q = einops.rearrange(q, "... l n d -> ... n l d")
            k = einops.rearrange(k, "... l n d -> ... n l d")
            v = einops.rearrange(v, "... l n d -> ... n l d")
            if not self.causal:
                if attention_mask is None:
                    attn_out = flex_attention(q, k, v)
                else:
                    attn_out = flex_attention(q, k, v, block_mask=attention_mask)
            else:
                assert (
                    attention_mask is not None
                ), "Causal attention requires attention_mask."
                attn_out = flex_attention(q, k, v, block_mask=attention_mask)
            attn_out = einops.rearrange(attn_out, "... n l d -> ... l (n d)")
        else:
            raise NotImplementedError(
                f"Attention implementation {self.attention_imp} not implemented."
            )

        attn_out = self.wo(attn_out)
        return attn_out


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_head,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        qk_norm: bool = False,
        causal: bool = False,
        bias: bool = True,
        use_checkpoint=False,
        attention_imp: AttentionImp = AttentionImp.FLASH_ATTENTION,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = MultiheadAttention(
            n_head,
            d_model,
            bias=bias,
            qk_norm=qk_norm,
            causal=causal,
            attention_imp=attention_imp,
        )
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, mlp_width)),
                        ("gelu", act_layer()),
                        ("c_proj", nn.Linear(mlp_width, d_model)),
                    ]
                )
            )
        self.use_checkpoint = use_checkpoint

    def attention(
        self,
        x: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        return self.attn(
            q=x, rope_freqs_cis=rope_freqs_cis, attention_mask=attention_mask
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if self.use_checkpoint and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(
                self._inner_forward,
                x,
                rope_freqs_cis,
                attention_mask,
                use_reentrant=False,
            )
        else:
            x = self._inner_forward(x, rope_freqs_cis, attention_mask)
        return x

    def _inner_forward(
        self,
        x: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        attn_output = self.attention(
            x=self.ln_1(x), rope_freqs_cis=rope_freqs_cis, attention_mask=attention_mask
        )
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    support_attention_mask_types = [
        AttentionMaskType.CAUSAL,
        AttentionMaskType.PREFIX_LM_CAUSAL,
        AttentionMaskType.FULL,
        AttentionMaskType.VIDEO_ENCODER_MASK,
    ]

    def __init__(
        self,
        *,
        image_size,
        image_channels,
        patch_size,
        model_size,
        num_latent_tokens,
        token_size,
        width: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        use_checkpoint=False,
        qk_norm: bool = False,
        causal=False,
        bias: bool = True,
        use_cls_token: bool = True,
        rope_layer: Rope3DPosEmb | None = None,
        positional_embedding_type: PositionalEmbedingType = PositionalEmbedingType.LEARNABLE,
        attention_imp: AttentionImp = AttentionImp.FLASH_ATTENTION,
        attention_mask_type: AttentionMaskType | None = None,
        temporal_size: int = 1,
        PFrame_tokens: int = 1,
        inside_latent_tokens: bool = False,
    ):
        super().__init__()
        # model config
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert len(image_size) == 2
            image_size = tuple(image_size)
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        grid_size = (
            self.temporal_size,
            image_size[0] // patch_size,
            image_size[1] // patch_size,
        )  # h, w
        self.grid_size = grid_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.tokens_pre_frame = int(np.prod(self.grid_size[1:]))
        self.PFrame_tokens = PFrame_tokens
        self.Iframe_tokens = (
            self.num_latent_tokens - (self.temporal_size - 1) * self.PFrame_tokens
        )
        self.token_size = token_size
        self.use_checkpoint = use_checkpoint
        self.width, self.num_layers, self.num_heads = TITOK_ARCH_DICT[model_size]
        if width is not None:
            self.width = width
        if num_layers is not None:
            self.num_layers = num_layers
        if num_heads is not None:
            self.num_heads = num_heads
        self.positional_embedding_type = positional_embedding_type

        scale = self.width**-0.5
        self.use_cls_token = use_cls_token
        self.causal = causal
        self.attention_imp = attention_imp
        self.attention_mask_type = attention_mask_type
        self.inside_latent_tokens = inside_latent_tokens
        if self.attention_mask_type is not None:
            assert (
                self.attention_mask_type in self.support_attention_mask_types
            ), f"Only support {self.support_attention_mask_types}"
            if self.attention_mask_type == AttentionMaskType.CAUSAL:
                if not causal:
                    logger.warning(
                        "Attention mask type is CAUSAL, but causal is False."
                    )
                    self.causal = True
                    causal = True
            else:
                if causal:
                    logger.warning(
                        "Attention mask type is not CAUSAL, but causal is True."
                    )
                    self.causal = False
                    causal = False

        # learnable model
        self.patch_embed = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        if self.use_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        if self.inside_latent_tokens:
            self.IFrame_latent_tokens = nn.Parameter(
                scale * torch.randn(self.Iframe_tokens, self.width)
            )
            self.PFrame_latent_tokens = None
            if self.temporal_size > 1:
                self.PFrame_latent_tokens = nn.Parameter(
                    scale * torch.randn(self.PFrame_tokens, self.width)
                )
        self.rope_layer = rope_layer
        if self.temporal_size > 1:
            assert (
                self.positional_embedding_type == PositionalEmbedingType.ROPE_3D
            ), "Only support 3D rope positional embedding for temporal size > 1."
            assert (
                not self.use_cls_token
            ), "Temporal size > 1 does not support cls token."
        if positional_embedding_type == PositionalEmbedingType.LEARNABLE:
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(int(np.prod(self.grid_size) + 1), self.width)
            )
            self.latent_token_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, self.width)
            )
        elif positional_embedding_type == PositionalEmbedingType.ROPE_3D:
            assert isinstance(
                self.rope_layer, Rope3DPosEmb
            ), "Rope layer must be provided for 3D rope positional embedding."
        else:
            raise NotImplementedError(
                f"Positional embedding type {positional_embedding_type} not implemented."
            )
        self.ln_pre = LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(
                    self.width,
                    self.num_heads,
                    mlp_ratio=4.0,
                    use_checkpoint=self.use_checkpoint,
                    bias=bias,
                    qk_norm=qk_norm,
                    causal=self.causal,
                    attention_imp=attention_imp,
                )
            )
        self.ln_post = LayerNorm(self.width)
        # self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)

        # extra cfg
        self.logged_attention_keys = set()

    @functools.cache
    def visul_token_len(self, forward_T):
        visual_grid_size = list(self.grid_size)
        if len(visual_grid_size) == 2:
            visual_grid_size = [forward_T] + visual_grid_size
        assert (
            len(visual_grid_size) == 3
        ), f"Expected 3D grid size, got {visual_grid_size}"
        visual_grid_size[0] = forward_T
        visual_token_len = int(np.prod(visual_grid_size)) + int(self.use_cls_token)
        return visual_token_len

    def latent_token_len(self, forward_T):
        num_latent_tokens = self.Iframe_tokens + (forward_T - 1) * self.PFrame_tokens
        return num_latent_tokens

    @functools.cache
    def seq_len(self, forward_T):
        seq_len = self.visul_token_len(forward_T) + self.latent_token_len(forward_T)
        return seq_len

    @functools.cache
    def get_attention_mask(self, forward_T: int):

        attention_mask = None
        mask_fn = None
        seq_len = self.seq_len(forward_T)
        if self.causal and self.attention_imp == AttentionImp.FLEX_ATTENTION:
            attention_mask = flex_attention_mod.create_block_mask(
                flex_attention_mask_mod.causal,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
            )
        elif self.attention_mask_type == AttentionMaskType.PREFIX_LM_CAUSAL:
            assert (
                self.attention_imp != AttentionImp.FLASH_ATTENTION
            ), "Flash Attention does not support attention_mask."
            # mask_fn = flex_attention_mask_mod.generate_prefix_lm_mask(prefix_len)
            mask_fn = flex_attention_mask_mod.VideoEncoderMask(
                num_frames=self.temporal_size,
                tokens_per_frame=self.tokens_pre_frame,
                IFrame_tokens=self.Iframe_tokens,
                PFrame_tokens=self.PFrame_tokens,
            )
            mask_fn = mask_fn
            if self.attention_imp == AttentionImp.FLEX_ATTENTION:
                attention_mask = flex_attention_mod.create_block_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
            elif self.attention_imp == AttentionImp.TORCH:
                attention_mask = flex_attention_mod.create_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
        elif self.attention_mask_type == AttentionMaskType.VIDEO_ENCODER_MASK:
            assert (
                self.attention_imp != AttentionImp.FLASH_ATTENTION
            ), "Flash Attention does not support attention_mask."
            mask_fn = flex_attention_mask_mod.VideoEncoderMask(
                num_frames=forward_T,
                tokens_per_frame=self.tokens_pre_frame,
                IFrame_tokens=self.Iframe_tokens,
                PFrame_tokens=self.PFrame_tokens,
            )
            assert seq_len == mask_fn.seq_len, f"{seq_len}, {mask_fn.seq_len}"
            if self.attention_imp == AttentionImp.FLEX_ATTENTION:
                attention_mask = flex_attention_mod.create_block_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
            elif self.attention_imp == AttentionImp.TORCH:
                attention_mask = flex_attention_mod.create_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
        return attention_mask, mask_fn

    @property
    def device(self):
        return next(self.parameters()).device

    @functools.cache
    def freqs_cis(self, forward_T: int):
        assert (
            self.rope_layer is not None
        ), "Rope layer must be provided for 3D rope positional embedding."
        rope_t_index = 0
        cls_rope_index = None
        if self.use_cls_token:
            cls_rope_index = Rope3DPosEmb.len_to_rope_index(1, device=self.device)
            cls_rope_index, rope_t_index = Rope3DPosEmb.shift_rope_index(
                cls_rope_index, rope_t_index, shift_all=True
            )
        # visual part
        visual_grad_size = list(self.grid_size)
        if len(visual_grad_size) == 2:
            visual_grad_size = [forward_T] + visual_grad_size
        assert (
            len(visual_grad_size) == 3
        ), f"Expected 3D grid size, got {visual_grad_size}"
        visual_grad_size[0] = forward_T
        visual_rope_index = Rope3DPosEmb.shape_to_index(
            *visual_grad_size, device=self.device
        )
        visual_rope_index, rope_t_index = Rope3DPosEmb.shift_rope_index(
            visual_rope_index, rope_t_index
        )
        # learnable query pos
        query_t_index = 0  # query t starts from 0
        query_len = self.Iframe_tokens + (forward_T - 1) * self.PFrame_tokens
        query_rope_index = Rope3DPosEmb.len_to_rope_index(query_len, device=self.device)
        query_rope_index, query_t_index = Rope3DPosEmb.shift_rope_index(
            query_rope_index, query_t_index, shift_all=True
        )
        if cls_rope_index is not None:
            rope_index = torch.cat(
                [cls_rope_index, visual_rope_index, query_rope_index], dim=0
            )
        else:
            rope_index = torch.cat([visual_rope_index, query_rope_index], dim=0)
        rope_index_mask = torch.ones_like(rope_index[..., 0], dtype=torch.bool)

        freqs_cis = self.rope_layer.get_freqs_cis_by_idx(rope_index, rope_index_mask)
        freqs_cis = freqs_cis.to(self.device)
        return freqs_cis

    def forward(self, pixel_values: torch.Tensor, latent_tokens=None, forward_T=None):
        if forward_T is None:
            forward_T = self.temporal_size
        if latent_tokens is None:
            if forward_T == 1:
                latent_tokens = self.IFrame_latent_tokens
            else:
                repeat_PFrame = einops.repeat(
                    self.PFrame_latent_tokens, "p d -> (t p) d", t=forward_T - 1
                )
                latent_tokens = torch.cat(
                    [self.IFrame_latent_tokens, repeat_PFrame], dim=0
                )
        # input_shape batch,c,h,w
        batch_size = pixel_values.shape[0]
        if pixel_values.dim() == 4:
            assert forward_T == 1, "Temporal size > 1 does not support 2D input."
            pixel_values = einops.rearrange(pixel_values, "b c h w -> b 1 c h w")
        elif pixel_values.dim() == 5:
            assert (
                forward_T == pixel_values.shape[1]
            ), "Temporal size does not match input shape."
        else:
            raise ValueError(f"Expected 4D or 5D input, got {pixel_values.shape}")
        x = einops.rearrange(pixel_values, "b t c h w -> (b t) c h w")
        x = self.patch_embed(x)
        x = einops.rearrange(x, "(b t) c h w -> b (t h w) c", b=batch_size)
        # class embeddings and positional embeddings
        if self.use_cls_token:
            x = torch.cat(
                [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1
            )
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)

        if self.positional_embedding_type == PositionalEmbedingType.LEARNABLE:
            x = x + self.positional_embedding.to(
                x.dtype
            )  # shape = [*, grid ** 2 + 1, width]
            latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(
                x.dtype
            )
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        attention_mask, _ = self.get_attention_mask(forward_T)
        for i in range(self.num_layers):
            if self.positional_embedding_type == PositionalEmbedingType.ROPE_3D:
                rope_freqs_cis = einops.repeat(
                    self.freqs_cis(forward_T), "n d -> b n d", b=batch_size
                )
                x = self.transformer[i](
                    x, rope_freqs_cis=rope_freqs_cis, attention_mask=attention_mask
                )
            else:
                x = self.transformer[i](x, attention_mask=attention_mask)

        latent_tokens = x[:, self.visul_token_len(forward_T) :]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = self.proj_out(latent_tokens)
        latent_tokens = einops.rearrange(
            latent_tokens, "b (h w) d -> b d h w", d=self.token_size, h=1
        )
        return latent_tokens


class TiTokDecoder(nn.Module):
    support_attention_mask_types = [
        AttentionMaskType.FULL,
        AttentionMaskType.VIDEO_DECODER_MASK,
    ]

    def __init__(
        self,
        *,
        image_size,
        image_channels,
        patch_size,
        model_size,
        num_latent_tokens,
        token_size,
        output_channels,
        width: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        use_checkpoint=False,
        qk_norm: bool = False,
        causal=False,
        bias: bool = True,
        code_drop: bool = False,
        use_cls_token: bool = True,
        rope_layer: Rope3DPosEmb | None = None,
        positional_embedding_type: PositionalEmbedingType = PositionalEmbedingType.LEARNABLE,
        attention_imp: AttentionImp = AttentionImp.FLASH_ATTENTION,
        attention_mask_type: AttentionMaskType | None = None,
        temporal_size: int = 1,
        PFrame_tokens: int = 1,
    ):
        super().__init__()
        # model cfg
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert len(image_size) == 2
            image_size = tuple(image_size)
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        grid_size = (
            self.temporal_size,
            image_size[0] // patch_size,
            image_size[1] // patch_size,
        )  # t, h, w
        self.grid_size = grid_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.tokens_pre_frame = int(np.prod(self.grid_size[1:]))
        self.PFrame_tokens = PFrame_tokens
        self.Iframe_tokens = (
            self.num_latent_tokens - (self.temporal_size - 1) * self.PFrame_tokens
        )
        self.token_size = token_size
        self.output_channels = output_channels
        self.use_checkpoint = use_checkpoint
        self.width, self.num_layers, self.num_heads = TITOK_ARCH_DICT[model_size]
        if width is not None:
            self.width = width
        if num_layers is not None:
            self.num_layers = num_layers
        if num_heads is not None:
            self.num_heads = num_heads
        self.positional_embedding_type = positional_embedding_type
        self.use_cls_token = use_cls_token
        self.causal = causal
        self.attention_imp = attention_imp
        self.attention_mask_type = attention_mask_type

        if self.attention_mask_type is not None:
            assert (
                self.attention_mask_type in self.support_attention_mask_types
            ), f"Only support {self.support_attention_mask_types}"
            if self.attention_mask_type == AttentionMaskType.CAUSAL:
                if not causal:
                    logger.warning(
                        "Attention mask type is CAUSAL, but causal is False."
                    )
                    self.causal = True
                    causal = True
            else:
                if causal:
                    logger.warning(
                        "Attention mask type is not CAUSAL, but causal is True."
                    )
                    self.causal = False
                    causal = False

        # learnable model
        self.rope_layer = rope_layer
        if self.temporal_size > 1:
            assert (
                self.positional_embedding_type == PositionalEmbedingType.ROPE_3D
            ), "Only support 3D rope positional embedding for temporal size > 1."
            assert (
                not self.use_cls_token
            ), "Temporal size > 1 does not support cls token."
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width**-0.5
        if self.use_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        if self.positional_embedding_type == PositionalEmbedingType.LEARNABLE:
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(int(np.prod(self.grid_size) + 1), self.width)
            )
            self.latent_token_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, self.width)
            )
        elif self.positional_embedding_type == PositionalEmbedingType.ROPE_3D:
            assert isinstance(
                self.rope_layer, Rope3DPosEmb
            ), "Rope layer must be provided for 3D rope positional embedding."
        else:
            raise NotImplementedError(
                f"Positional embedding type {positional_embedding_type} not implemented."
            )
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(
                    self.width,
                    self.num_heads,
                    mlp_ratio=4.0,
                    use_checkpoint=self.use_checkpoint,
                    bias=bias,
                    qk_norm=qk_norm,
                    causal=self.causal,
                    attention_imp=attention_imp,
                )
            )
        self.ln_post = LayerNorm(self.width)

        self.ffn = nn.Sequential(
            # nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
            nn.Linear(self.width, 2 * self.width, bias=True),
            nn.Tanh(),
            # nn.Conv2d(2 * self.width, self.output_channels, 1, padding=0, bias=True),
            nn.Linear(2 * self.width, self.output_channels, bias=True),
        )

        self.conv_out = nn.Identity()
        self.code_drop = code_drop

        # extra cfg
        self.logged_attention_keys = set()

    @functools.cache
    def visul_token_len(self, forward_T):
        visual_grid_size = list(self.grid_size)
        if len(visual_grid_size) == 2:
            visual_grid_size = [forward_T] + visual_grid_size
        assert (
            len(visual_grid_size) == 3
        ), f"Expected 3D grid size, got {visual_grid_size}"
        visual_grid_size[0] = forward_T
        visual_token_len = int(np.prod(visual_grid_size)) + int(self.use_cls_token)
        return visual_token_len

    def latent_token_len(self, forward_T):
        num_latent_tokens = self.Iframe_tokens + (forward_T - 1) * self.PFrame_tokens
        return num_latent_tokens

    @functools.cache
    def seq_len(self, forward_T):
        seq_len = self.visul_token_len(forward_T) + self.latent_token_len(forward_T)
        return seq_len

    @functools.cache
    def get_attention_mask(self, forward_T: int):
        attention_mask = None
        mask_fn = None
        if self.attention_mask_type == AttentionMaskType.VIDEO_DECODER_MASK:
            assert (
                self.attention_imp != AttentionImp.FLASH_ATTENTION
            ), "Flash Attention does not support attention_mask."
            mask_fn = flex_attention_mask_mod.VideoDecoderMask(
                num_frames=forward_T,
                tokens_per_frame=self.tokens_pre_frame,
                IFrame_tokens=self.Iframe_tokens,
                PFrame_tokens=self.PFrame_tokens,
            )
            seq_len = self.seq_len(forward_T)
            assert seq_len == mask_fn.seq_len, f"{seq_len}, {mask_fn.seq_len}"
            if self.attention_imp == AttentionImp.FLEX_ATTENTION:
                attention_mask = flex_attention_mod.create_block_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
            elif self.attention_imp == AttentionImp.TORCH:
                attention_mask = flex_attention_mod.create_mask(
                    mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                )
        return attention_mask, mask_fn

    @property
    def device(self):
        return next(self.parameters()).device

    @functools.cache
    def freqs_cis(self, forward_T: int):
        assert (
            self.rope_layer is not None
        ), "Rope layer must be provided for 3D rope positional embedding."
        rope_t_index = 0
        cls_rope_index = None
        if self.use_cls_token:
            cls_rope_index = Rope3DPosEmb.len_to_rope_index(1, device=self.device)
            cls_rope_index, rope_t_index = Rope3DPosEmb.shift_rope_index(
                cls_rope_index, rope_t_index, shift_all=True
            )
        # visual part
        visual_grad_size = list(self.grid_size)
        if len(visual_grad_size) == 2:
            visual_grad_size = [forward_T] + visual_grad_size
        assert (
            len(visual_grad_size) == 3
        ), f"Expected 3D grid size, got {visual_grad_size}"
        visual_grad_size[0] = forward_T
        visual_rope_index = Rope3DPosEmb.shape_to_index(
            *visual_grad_size, device=self.device
        )
        visual_rope_index, rope_t_index = Rope3DPosEmb.shift_rope_index(
            visual_rope_index, rope_t_index
        )
        # learnable query pos
        query_t_index = 0  # query t starts from 0
        query_len = self.Iframe_tokens + (forward_T - 1) * self.PFrame_tokens
        query_rope_index = Rope3DPosEmb.len_to_rope_index(query_len, device=self.device)
        query_rope_index, query_t_index = Rope3DPosEmb.shift_rope_index(
            query_rope_index, query_t_index, shift_all=True
        )
        if cls_rope_index is not None:
            visual_rope_index = torch.cat([cls_rope_index, visual_rope_index], dim=0)
        rope_index = torch.cat([visual_rope_index, query_rope_index], dim=0)
        rope_index_mask = torch.ones_like(rope_index[..., 0], dtype=torch.bool)

        freqs_cis = self.rope_layer.get_freqs_cis_by_idx(rope_index, rope_index_mask)
        freqs_cis = freqs_cis.to(self.device)
        visual_freqs_cis = freqs_cis[: visual_rope_index.shape[0]]
        query_freqs_cis = freqs_cis[visual_rope_index.shape[0] :]
        return visual_freqs_cis, query_freqs_cis

    def forward(self, z_quantized, forward_T=None):
        if forward_T is None:
            forward_T = self.temporal_size
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.latent_token_len(
            forward_T
        ), f"{H}, {W}, {self.latent_token_len(forward_T)}"
        x = z_quantized.reshape(N, C * H, W).permute(0, 2, 1)  # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(
            batchsize, self.visul_token_len(forward_T), 1
        ).to(x.dtype)
        if self.use_cls_token:
            mask_tokens = torch.cat(
                [
                    _expand_token(self.class_embedding, mask_tokens.shape[0]).to(
                        mask_tokens.dtype
                    ),
                    mask_tokens,
                ],
                dim=1,
            )
        visual_freqs_cis, query_freqs_cis = None, None
        if self.positional_embedding_type == PositionalEmbedingType.LEARNABLE:
            mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
            x = x + self.latent_token_positional_embedding[:seq_len]
        elif self.positional_embedding_type == PositionalEmbedingType.ROPE_3D:
            visual_freqs_cis, query_freqs_cis = self.freqs_cis(forward_T)
        if self.code_drop:
            # 随机丢弃部分[0,n-1]个latent token
            used_nums_latent_tokens = torch.randint(
                1, self.num_latent_tokens + 1, (1,)
            ).item()
            x = x[:, :used_nums_latent_tokens]
            query_freqs_cis = (
                query_freqs_cis[:used_nums_latent_tokens]
                if query_freqs_cis is not None
                else None
            )
        if query_freqs_cis is not None and visual_freqs_cis is not None:
            freqs_cis = torch.cat([visual_freqs_cis, query_freqs_cis], dim=0)
            freqs_cis = einops.repeat(freqs_cis, "n d -> b n d", b=batchsize)
        x = torch.cat([mask_tokens, x], dim=1)
        x = self.ln_pre(x)
        attention_mask, _ = self.get_attention_mask(forward_T)
        for i in range(self.num_layers):
            if self.positional_embedding_type == PositionalEmbedingType.ROPE_3D:
                x = self.transformer[i](
                    x, rope_freqs_cis=freqs_cis, attention_mask=attention_mask
                )
            else:
                x = self.transformer[i](x, attention_mask=attention_mask)
        shift = int(self.use_cls_token)
        x = x[:, shift : self.visul_token_len(forward_T)]
        x = self.ln_post(x)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        # N L D -> N D H W
        x = einops.rearrange(
            x,
            "n (t h w) c -> (n t) c h w",
            t=forward_T,
            h=self.grid_size[1],
            w=self.grid_size[2],
        )
        if forward_T > 1:
            x = einops.rearrange(x, "(n t) c h w -> n t c h w", t=forward_T)
        return x
