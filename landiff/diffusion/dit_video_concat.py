from functools import partial

import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from sat.helpers import print_rank0
from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.mpu.layers import ColumnParallelLinear
from sat.ops.layernorm import LayerNorm, RMSNorm
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from torch import nn

from landiff.diffusion.sgm.modules.diffusionmodules.openaimodel import Timestep
from landiff.diffusion.sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
from landiff.diffusion.sgm.util import InferValueRegistry, instantiate_from_config
from landiff.utils import freeze_model, zero_module


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"]  # (b,t,c,h,w)
        B, T = images.shape[:2]
        emb = images.view(-1, *images.shape[2:])
        emb = self.proj(emb)  # ((b t),d,h/2,w/2)
        emb = emb.view(B, T, *emb.shape[1:])
        emb = emb.flatten(3).transpose(2, 3)  # (b,t,n,d)
        emb = rearrange(emb, "b t n d -> b (t n) d")

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d)

        emb = emb.contiguous()
        return emb  # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_height * grid_width, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    return pos_embed  # [T, H*W, D]


def get_2d_sincos_pos_embed(
    embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self, height, width, compressed_num_frames, hidden_size, text_length=0
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)),
            requires_grad=False,
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embedding.shape[-1], self.height, self.width
        )
        self.pos_embedding.data[:, -self.spatial_length :].copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )


class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.text_length = text_length
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)),
            requires_grad=False,
        )
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs["images"].shape[1] == 1:
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
        learnable_pos_embed=False,
    ):
        super().__init__()
        self.rot_v = rot_v

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        freqs_t = 1.0 / (
            theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
        )
        freqs_h = 1.0 / (
            theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
        )
        freqs_w = 1.0 / (
            theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
        )

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat(
            (
                freqs_t[:, None, None, :],
                freqs_h[None, :, None, :],
                freqs_w[None, None, :, :],
            ),
            dim=-1,
        )
        freqs = rearrange(freqs, "t h w d -> (t h w) d")

        freqs = freqs.contiguous()
        freqs_sin = freqs.sin()
        freqs_cos = freqs.cos()
        self.register_buffer("freqs_sin", freqs_sin)
        self.register_buffer("freqs_cos", freqs_cos)

        self.text_length = text_length
        if learnable_pos_embed:
            num_patches = height * width * compressed_num_frames + text_length
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True
            )
        else:
            self.pos_embedding = None

    def rotary(self, t, **kwargs):
        seq_len = t.shape[2]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]

        query_layer[:, :, self.text_length :] = self.rotary(
            query_layer[:, :, self.text_length :]
        )
        key_layer[:, :, self.text_length :] = self.rotary(
            key_layer[:, :, self.text_length :]
        )
        if self.rot_v:
            value_layer[:, :, self.text_length :] = self.rotary(
                value_layer[:, :, self.text_length :]
            )

        return attention_fn_default(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum("nlpqc->ncplq", x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(
            x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p
        )

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=elementwise_affine, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
        )

        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)
            ]
        )

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args["layer_id"]].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args["layer_id"]](x)
        hidden = origin.activation_func(x2) * x1
        x = origin.dense_4h_to_h(hidden)
        return x


class AdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.hidden_size = hidden_size

        self.adaLN_modulations = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size))
                for _ in range(num_layers)
            ]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(
                        hidden_size_head,
                        eps=1e-6,
                        elementwise_affine=elementwise_affine,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(
                        hidden_size_head,
                        eps=1e-6,
                        elementwise_affine=elementwise_affine,
                    )
                    for _ in range(num_layers)
                ]
            )

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)

        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(
            text_attention_input, text_shift_msa, text_scale_msa
        )

        attention_input = torch.cat(
            (text_attention_input, img_attention_input), dim=1
        )  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)
        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = (
            img_hidden_states + gate_msa * img_attention_output
        )  # (b,(t n),d)
        text_hidden_states = (
            text_hidden_states + text_gate_msa * text_attention_output
        )  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(
            img_hidden_states
        )  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(
            text_hidden_states
        )  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat(
            (text_mlp_input, img_mlp_input), dim=1
        )  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = (
            img_hidden_states + gate_mlp * img_mlp_output
        )  # vision (b,(t n),d)
        text_hidden_states = (
            text_hidden_states + text_gate_mlp * text_mlp_output
        )  # language (b,n,d)

        hidden_states = torch.cat(
            (text_hidden_states, img_hidden_states), dim=1
        )  # (b,(n_t+t*n_i),d)
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class DiffusionTransformer(BaseModel):
    def __init__(
        self,
        transformer_args,
        num_frames,
        time_compressed_rate,
        latent_width,
        latent_height,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_layers,
        num_attention_heads,
        elementwise_affine,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time="adaln",
        adm_in_channels=None,
        parallel_output=True,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        use_SwiGLU=False,
        use_RMSNorm=False,
        zero_init_y_embed=False,
        **kwargs,
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.time_embed_dim = (
            time_embed_dim if time_embed_dim is not None else hidden_size
        )
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        self.inner_hidden_size = hidden_size * 4
        self.zero_init_y_embed = zero_init_y_embed
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32

        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            kwargs["layernorm"] = partial(
                LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6
            )

        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        module_configs = modules
        self._build_modules(module_configs)

        if use_SwiGLU:
            self.add_mixin(
                "swiglu",
                SwiGLUMixin(
                    num_layers, hidden_size, self.inner_hidden_size, bias=False
                ),
                reinit=True,
            )

    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
                if self.zero_init_y_embed:
                    nn.init.constant_(self.label_emb[0][2].weight, 0)
                    nn.init.constant_(self.label_emb[0][2].bias, 0)
            else:
                raise ValueError()

        pos_embed_config = module_configs["pos_embed_config"]
        self.add_mixin(
            "pos_embed",
            instantiate_from_config(
                pos_embed_config,
                height=self.latent_height // self.patch_size,
                width=self.latent_width // self.patch_size,
                compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate
                + 1,
                hidden_size=self.hidden_size,
            ),
            reinit=True,
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.add_mixin(
            "patch_embed",
            instantiate_from_config(
                patch_embed_config,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                in_channels=self.in_channels,
            ),
            reinit=True,
        )
        if self.input_time == "adaln":
            adaln_layer_config = module_configs["adaln_layer_config"]
            self.add_mixin(
                "adaln_layer",
                instantiate_from_config(
                    adaln_layer_config,
                    height=self.latent_height // self.patch_size,
                    width=self.latent_width // self.patch_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    compressed_num_frames=(self.num_frames - 1)
                    // self.time_compressed_rate
                    + 1,
                    hidden_size_head=self.hidden_size // self.num_attention_heads,
                    time_embed_dim=self.time_embed_dim,
                    elementwise_affine=self.elementwise_affine,
                ),
            )
        else:
            raise NotImplementedError

        final_layer_config = module_configs["final_layer_config"]
        self.add_mixin(
            "final_layer",
            instantiate_from_config(
                final_layer_config,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                time_embed_dim=self.time_embed_dim,
                latent_width=self.latent_width,
                latent_height=self.latent_height,
                elementwise_affine=self.elementwise_affine,
            ),
            reinit=True,
        )

        if "lora_config" in module_configs:
            lora_config = module_configs["lora_config"]
            self.add_mixin(
                "lora",
                instantiate_from_config(lora_config, layer_num=self.num_layers),
                reinit=True,
            )

        return

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # This is not use in inference
        if "concat_images" in kwargs and kwargs["concat_images"] is not None:
            if kwargs["concat_images"].shape[0] != x.shape[0]:
                concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
            else:
                concat_images = kwargs["concat_images"]
            x = torch.cat([x, concat_images], dim=2)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False, dtype=self.dtype
        )
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs["seq_length"] = t * h * w // (self.patch_size**2)
        kwargs["images"] = x
        kwargs["emb"] = emb
        kwargs["encoder_outputs"] = context
        kwargs["text_length"] = context.shape[1]

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = (
            torch.ones((1, 1)).to(x.dtype)
        )
        output = super().forward(**kwargs)[0]
        return output


class ControlDiffusionTransformer(DiffusionTransformer):
    def __init__(self, *args, **kwargs):
        self.use_semantic_injection_adaln = kwargs.pop(
            "use_semantic_injection_adaln", False
        )
        self.use_uncertainty_sampling = kwargs.pop("uncertainty_sampling_mode", "auto")
        self.semantic_video_frames = kwargs.pop("semantic_video_frames", "auto")
        super().__init__(*args, **kwargs)

    def _build_modules(self, module_configs):
        super()._build_modules(module_configs)

        # budil feature extractor
        semantic_condition_config = module_configs["semantic_condition_config"]
        self.semantic_conditioner = instantiate_from_config(
            semantic_condition_config, dtype=self.dtype
        )
        if self.use_semantic_injection_adaln:
            self.semantic_injection_adaln_layer = SemanticInjectionAdaLN(
                hidden_size=self.in_channels,
                uncertainty_sampling_mode=self.use_uncertainty_sampling,
            )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if InferValueRegistry.get_value("semantic_feature") is None or self.training:
            semantic_token = InferValueRegistry.get_value("semantic_token")
            semantic_feature_before_upsample = InferValueRegistry.get_value(
                "semantic_feature_before_upsample"
            )
            if semantic_token is not None:
                semantic_feature = self.semantic_conditioner(indexs=semantic_token)
            elif semantic_feature_before_upsample is not None:
                semantic_feature = self.semantic_conditioner(
                    semantic_feature_before_upsample=semantic_feature_before_upsample
                )
            else:
                input_video = kwargs["mp4"]
                if isinstance(self.semantic_video_frames, int):
                    assert (
                        self.semantic_video_frames <= input_video.shape[1]
                    ), f"semantic_video_frames should be less than or equal to input_video.shape[1], got {self.semantic_video_frames} and {input_video.shape[1]}"
                    sample_idx = (
                        torch.linspace(
                            0, input_video.shape[1] - 1, self.semantic_video_frames
                        )
                        .long()
                        .to(x.device)
                    )
                else:
                    # 等间距采样到t
                    sample_idx = (
                        torch.linspace(0, input_video.shape[1] - 1, t)
                        .long()
                        .to(x.device)
                    )
                input_video = input_video[:, sample_idx, ...]
                semantic_feature = self.semantic_conditioner(
                    input_video, vq_origin_features=kwargs.get("vae_featrues", None)
                )
            if semantic_feature.dtype != self.dtype:
                semantic_feature = semantic_feature.to(self.dtype)
            if not self.training:
                InferValueRegistry.register("semantic_feature", semantic_feature)
        else:
            semantic_feature = InferValueRegistry.get_value("semantic_feature")
            assert semantic_feature is not None
            if semantic_feature.dtype != self.dtype:
                semantic_feature = semantic_feature.to(self.dtype)
        if self.use_semantic_injection_adaln:
            x = einops.rearrange(x, "b t c h w -> b t h w c")
            semantic_feature = einops.rearrange(
                semantic_feature, "b t c h w -> b t h w c"
            )
            x = self.semantic_injection_adaln_layer(x, semantic_feature)
            x = einops.rearrange(x, "b t h w c -> b t c h w")
        else:
            x = x + semantic_feature
        # This is not use in inference
        if "concat_images" in kwargs and kwargs["concat_images"] is not None:
            if kwargs["concat_images"].shape[0] != x.shape[0]:
                concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
            else:
                concat_images = kwargs["concat_images"]
            x = torch.cat([x, concat_images], dim=2)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False, dtype=self.dtype
        )
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs["seq_length"] = t * h * w // (self.patch_size**2)
        kwargs["images"] = x
        kwargs["emb"] = emb
        kwargs["encoder_outputs"] = context
        kwargs["text_length"] = context.shape[1]

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = (
            torch.ones((1, 1)).to(x.dtype)
        )
        output = super(DiffusionTransformer, self).forward(
            **kwargs, output_hidden_states=True
        )
        output = output[1:]
        return output


class ControlMLPAdapter(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        in_channels: int,
        patch_size: int,
        use_zero_linears: bool,
        module_configs={},
        semantic_video_frames: int | str = "auto",
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_zero_linears = use_zero_linears
        self.module_configs = module_configs
        self.semantic_video_frames = semantic_video_frames

        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32

        self._build_modules(self.module_configs)

    def _build_modules(self, module_configs):
        # budil feature extractor
        semantic_condition_config = module_configs["semantic_condition_config"]
        self.semantic_conditioner = instantiate_from_config(
            semantic_condition_config, dtype=self.dtype
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.patch_embed: ImagePatchEmbeddingMixin = instantiate_from_config(
            patch_embed_config,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            in_channels=self.in_channels,
        )

        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size * 2),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                )
                for _ in range(self.num_layers)
            ]
        )
        if self.use_zero_linears:
            self.zero_linears = zero_module(
                nn.ModuleList(
                    [
                        nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                        for _ in range(self.num_layers)
                    ]
                )
            )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if InferValueRegistry.get_value("semantic_feature") is None or self.training:
            semantic_token = InferValueRegistry.get_value("semantic_token")
            if semantic_token is not None:
                semantic_feature = self.semantic_conditioner(indexs=semantic_token)
            else:
                input_video = kwargs["mp4"]
                if isinstance(self.semantic_video_frames, int):
                    assert (
                        self.semantic_video_frames <= input_video.shape[1]
                    ), f"semantic_video_frames should be less than or equal to input_video.shape[1], got {self.semantic_video_frames} and {input_video.shape[1]}"
                    sample_idx = (
                        torch.linspace(
                            0, input_video.shape[1] - 1, self.semantic_video_frames
                        )
                        .long()
                        .to(x.device)
                    )
                else:
                    # 等间距采样到t
                    sample_idx = (
                        torch.linspace(0, input_video.shape[1] - 1, t)
                        .long()
                        .to(x.device)
                    )
                input_video = input_video[:, sample_idx, ...]
                semantic_feature = self.semantic_conditioner(
                    input_video, vq_origin_features=kwargs.get("vae_featrues", None)
                )
            if semantic_feature.dtype != self.dtype:
                semantic_feature = semantic_feature.to(self.dtype)
            if not self.training:
                InferValueRegistry.register("semantic_feature", semantic_feature)
        else:
            semantic_feature = InferValueRegistry.get_value("semantic_feature")
            assert semantic_feature is not None
            if semantic_feature.dtype != self.dtype:
                semantic_feature = semantic_feature.to(self.dtype)
        hidden = self.patch_embed.word_embedding_forward(
            images=x, input_ids=None, encoder_outputs=context
        )
        hiddens = []
        for adapter in self.adapters:
            if self.training and hidden.requires_grad:
                hidden = (
                    torch.utils.checkpoint.checkpoint(
                        adapter, hidden, use_reentrant=False
                    )
                    + hidden
                )
            else:
                hidden = adapter(hidden) + hidden
            hiddens.append(hidden)
        if self.use_zero_linears:
            outputs = []
            for hidden, zero_linear in zip(hiddens, self.zero_linears):
                out = zero_linear(hidden)
                outputs.append(out)
            results = outputs
        else:
            results = hiddens
        if not self.training:
            InferValueRegistry.register("cache_control_output", results)
        return results


class ControlDiffWarp(nn.Module):
    def __init__(
        self,
        main_model: DiffusionTransformer,
        control_model: ControlDiffusionTransformer | ControlMLPAdapter,
        pretrain_diffusion_model_ckpt_path: str,
        freeze_dit: bool,
    ):
        super().__init__()
        self.main_model = main_model
        self.control_model = control_model
        self.freeze_dit = freeze_dit
        static = torch.load(pretrain_diffusion_model_ckpt_path, map_location="cpu")
        static = static["module"]
        new_staic = {}
        for key, value in static.items():
            if key.startswith("model."):
                new_staic[key[6:]] = value
        missing_keys, unexpected_keys = self.main_model.load_state_dict(
            new_staic, strict=False
        )
        assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
        if isinstance(control_model, ControlDiffusionTransformer):
            missing_keys, unexpected_keys = self.control_model.load_state_dict(
                new_staic, strict=False
            )
        # assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
        if self.freeze_dit:
            freeze_model(self.main_model)

        print_rank0("No training data specified", level="WARNING")

    def forward(self, *args, **kwargs):
        control_layers_output = self.control_model(*args, **kwargs)
        kwargs["control_layers_output"] = control_layers_output
        output = self.main_model(*args, **kwargs)
        return output


class ControlOutAdaLNMixin(AdaLNMixin):
    def __init__(self, *args, **kwargs):
        self.use_zero_linears = kwargs.pop("use_zero_linears", True)
        self.augmenter_params = kwargs.pop("augmenter_params", None)
        super().__init__(*args, **kwargs)
        # add zero linear layer for control
        if self.use_zero_linears:
            self.zero_linears = zero_module(
                nn.ModuleList(
                    [
                        nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                        for _ in range(self.num_layers)
                    ]
                )
            )

        if self.augmenter_params is not None:
            self.augmenter = NormalAugmenter(**self.augmenter_params)
        else:
            self.augmenter = None

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        hidden_states = super().layer_forward(hidden_states, mask, *args, **kwargs)
        if self.augmenter is not None:
            hidden_states = self.augmenter(hidden_states)
        if self.use_zero_linears:
            layer_id = int(kwargs["layer_id"])
            linear_layer = self.zero_linears[layer_id]
            hidden_states = linear_layer(hidden_states)
        return hidden_states


class ControlAdaLNMixin(AdaLNMixin):
    def __init__(self, *args, **kwargs):
        self.use_semantic_injection_adaln = kwargs.pop(
            "use_semantic_injection_adaln", False
        )
        self.control_layers = kwargs.pop("control_layers", 15)
        self.use_uncertainty_sampling = kwargs.pop("uncertainty_sampling_mode", "auto")
        super().__init__(*args, **kwargs)
        if self.use_semantic_injection_adaln:
            self.semantic_injection_adaln_layers = nn.ModuleList(
                [
                    SemanticInjectionAdaLN(
                        hidden_size=self.hidden_size,
                        uncertainty_sampling_mode=self.use_uncertainty_sampling,
                    )
                    for _ in range(self.control_layers)
                ]
            )

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]

        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)

        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(
            text_attention_input, text_shift_msa, text_scale_msa
        )

        attention_input = torch.cat(
            (text_attention_input, img_attention_input), dim=1
        )  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)
        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = (
            img_hidden_states + gate_msa * img_attention_output
        )  # (b,(t n),d)
        text_hidden_states = (
            text_hidden_states + text_gate_msa * text_attention_output
        )  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(
            img_hidden_states
        )  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(
            text_hidden_states
        )  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat(
            (text_mlp_input, img_mlp_input), dim=1
        )  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = (
            img_hidden_states + gate_mlp * img_mlp_output
        )  # vision (b,(t n),d)
        text_hidden_states = (
            text_hidden_states + text_gate_mlp * text_mlp_output
        )  # language (b,n,d)

        hidden_states = torch.cat(
            (text_hidden_states, img_hidden_states), dim=1
        )  # (b,(n_t+t*n_i),d)

        # control video code start
        control_layers_output = kwargs["control_layers_output"]
        layer_id = int(kwargs["layer_id"])
        assert (
            len(control_layers_output) == self.control_layers
        ), f"{len(control_layers_output)} != {self.control_layers}"
        if layer_id < len(control_layers_output):
            if isinstance(control_layers_output[layer_id], torch.Tensor):
                control_output = control_layers_output[layer_id]
                hidden_states[:, text_length:] = (
                    hidden_states[:, text_length:] + control_output
                )
            else:
                control_output = control_layers_output[layer_id]["hidden_states"]
                if self.use_semantic_injection_adaln:
                    hidden_states = self.semantic_injection_adaln_layers[layer_id](
                        hidden_states, control_output
                    )
                else:
                    hidden_states = hidden_states + control_output

        return hidden_states


class EmptyFinalLayerMixin(BaseMixin):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def final_forward(self, *args, **kwargs):
        return None

    def reinit(self, parent_model=None):
        pass
