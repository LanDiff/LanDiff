# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from itertools import chain
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn.functional import interpolate
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTModel


def handle_feature_output(
    x: torch.Tensor,
    feature_reduce_method: str | None = None,
    num_discard_tokens: int = 0,
) -> torch.Tensor:
    """Handle feature output from transformer.

    Args:
        x (torch.Tensor): input feature to be handled. shape is
            [B, 1+H*W+N, C] if including both CLS and register tokens.
            [B, 1+H*W, C] for standard model (N=0).
            [B, H*W, C] for model without CLS.
        feature_reduce_method (Optional[str]): method to select token. Options:
            - `mean_pooling`: average over spatial tokens (non CLS tokens), output shape = [B, C].
            - `max_pooling`: max over spatial tokens, output shape = [B, C].
            - `cls`: return CLS token only, output shape = [B, C].
            - `identity`: return the feature without touching it, output shape = input shape.
            - `None`: return spatial tokens, output shape = [B, H*W, C] (assuming input is [B, 1+H*W, C]).
            suppose raw feature is in shape [B, 1+H*W, C], `1` corresponds to CLS token.
        num_discard_tokens (int):
            number of tokens to be discarded. Assuming they are at the end of the sequence.
    Returns:
        torch.Tensor: selected feature tokens.
    """

    match feature_reduce_method:
        case "mean_pooling":
            return torch.mean(x[:, 1 : x.size(1) - num_discard_tokens], dim=1)  # [B, C]
        case "max_pooling":
            return torch.amax(x[:, 1 : x.size(1) - num_discard_tokens], dim=1)  # [B, C]
        case "cls":
            return x[:, 0]  # [B, C]
        case "identity":
            return x
        case None:
            return x[:, 1 : x.size(1) - num_discard_tokens]
        case _:
            raise NotImplementedError(
                f"feature_reduce_method {feature_reduce_method} it not implemented."
            )


# Modified from huggingface transformers ViTEmbeddings
# Original Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class ViTEmbeddingsNoCLS(ViTEmbeddings):
    """ViT Embedding Module without CLS token."""

    def __init__(self, config: AutoConfig, use_mask_token: bool = False):
        """Initialization.

        Args:
            config (AutoConfig): config for ViT.
            use_mask_token (bool, optional): whether to use mask token. Defaults to False.
        """
        super().__init__(config, use_mask_token=use_mask_token)
        self.cls_token = None

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings[:, 1:]

        embeddings = self.dropout(embeddings)

        return embeddings


# modified from huggingface transformers ViTModel
class ViTModelNoCLS(ViTModel):
    """ViT Model without CLS token."""

    def __init__(
        self,
        config: AutoConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = ViTEmbeddingsNoCLS(config, use_mask_token=use_mask_token)
        self.no_cls = True

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


# modified from huggingface transformers ViTEmbeddings
class ViTEmbeddingsReg(ViTEmbeddings):
    """ViT Embedding Module with register tokens.

    https://openreview.net/forum?id=2dnO3LLiJ1
    """

    def __init__(
        self, config: AutoConfig, use_mask_token: bool = False, num_reg_tokens: int = 7
    ):
        super().__init__(config, use_mask_token=use_mask_token)
        self.reg_token = nn.Parameter(
            torch.randn(1, num_reg_tokens, config.hidden_size)
        )
        self.num_reg_tokens = num_reg_tokens
        self.reg_pos_embed = nn.Parameter(
            torch.randn(1, num_reg_tokens, config.hidden_size)
        )

        self.reg_pos_embed.data = nn.init.trunc_normal_(
            self.reg_pos_embed.data.to(torch.float32),
            mean=0.0,
            std=self.config.initializer_range,
        ).to(self.reg_pos_embed.dtype)

        self.reg_token.data = nn.init.trunc_normal_(
            self.reg_token.data.to(torch.float32),
            mean=0.0,
            std=self.config.initializer_range,
        ).to(self.reg_token.dtype)

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1 - self.num_reg_tokens
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        reg_pos_embed = self.reg_pos_embed
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed, reg_pos_embed), dim=1
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        reg_tokens = self.reg_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings, reg_tokens), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + torch.cat(
                [self.position_embeddings, self.reg_pos_embed], dim=1
            )

        embeddings = self.dropout(embeddings)

        return embeddings


# modified from huggingface transformers ViTModel
class ViTModelReg(ViTModel):
    """ViT Model with register tokens.

    https://openreview.net/forum?id=2dnO3LLiJ1
    """

    def __init__(
        self,
        config: AutoConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
        num_reg_tokens: int = 7,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = ViTEmbeddingsReg(
            config, use_mask_token=use_mask_token, num_reg_tokens=num_reg_tokens
        )
        self.num_reg_tokens = num_reg_tokens

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


class DeiT(nn.Module):
    """DeiT model.

    Paper: Training data-efficient image transformers & distillation through attention
        https://arxiv.org/abs/2012.12877
    Huggingface Reference: https://huggingface.co/docs/transformers/en/model_doc/deit

    Attributes:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size
        model = AutoModel.from_pretrained(model_name)
        if pretrained:
            self.model = model
        else:
            deit_config = model.config
            self.model = AutoModel.from_config(deit_config)
            del model

        self.model.pooler = nn.Identity()

        # self.processor = AutoProcessor.from_pretrained(model_name)

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (224, 224)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)[:, 1:]  # for getting feature size, discard cls token
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: bool | None = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

        Returns:
            torch.Tensor: model output.
        """
        # input = self.processor(
        #     x, return_tensors="pt", do_resize=do_resize, do_rescale=do_rescale, do_normalize=do_normalize)
        input = self.yax_processor(x)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state

    def yax_processor(self, x):
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = (x - 127.5) / 127.5
        x = {
            "pixel_values": x,
        }
        return x


class DeiTNoCLS(nn.Module):
    """Modified DeiT model without CLS token."""

    def __init__(
        self,
        model_name: str = "nocls-facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size
        pretrained_model_name = model_name.replace("nocls-", "")
        deit_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.model = ViTModelNoCLS(deit_config)
        if pretrained:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
            pretrained_dict = {
                k: v
                for k, v in pretrained_model.state_dict().items()
                if k in self.model.state_dict()
            }
            self.load_state_dict(pretrained_dict, strict=False)
            del pretrained_model, pretrained_dict

        self.model.pooler = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.no_cls = True

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (self.image_size, self.image_size)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: bool | None = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

            - argument for forward
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.

        Returns:
            torch.Tensor: model output.
        """
        input = self.processor(
            x,
            return_tensors="pt",
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        ).to(self.model.device)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state


class DeiTReg(nn.Module):
    """Modified DeiT model with register tokens."""

    def __init__(
        self,
        model_name: str = "reg-facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
        num_reg_tokens: int = 7,
    ):
        super().__init__()
        self.image_size = image_size
        pretrained_model_name = model_name.replace("reg-", "")
        deit_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.model = ViTModelReg(deit_config, num_reg_tokens=num_reg_tokens)
        if pretrained:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
            pretrained_dict = {
                k: v
                for k, v in pretrained_model.state_dict().items()
                if k in self.model.state_dict()
            }
            self.load_state_dict(pretrained_dict, strict=False)
            del pretrained_model, pretrained_dict

        self.model.pooler = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.num_reg_tokens = num_reg_tokens

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (self.image_size, self.image_size)
            x = torch.zeros((1, *image_size, 3), dtype=torch.uint8)
            y = self.forward(x)[:, 1 : -self.num_reg_tokens]
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: bool | None = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

        Returns:
            torch.Tensor: model output.
        """
        input = self.processor(
            x,
            return_tensors="pt",
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        ).to(self.model.device)
        y = self.model(**input, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state


def build_backbone(
    model_name: str, pretrained: bool = False, image_size: int = 224, **kwargs: Any
) -> nn.Module:
    """Build the backbone visual encoder of robot vision foundation model.

    Args:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights. Defaults to False.
        image_size (int): size of the image. Assume a square image. Defaults to 224
        kwargs (Any): any kwargs specific to some models. For example,
            `num_reg_tokens` for `DeiTReg` when `"reg"` in `model_name`

    Returns:
        nn.Module: backbone network.
    """
    if "reg" in model_name:
        return DeiTReg(
            model_name=model_name,
            pretrained=pretrained,
            image_size=image_size,
            **kwargs,
        )
    elif "nocls" in model_name:
        return DeiTNoCLS(
            model_name=model_name,
            pretrained=pretrained,
            image_size=image_size,
            **kwargs,
        )
    elif "deit" in model_name:
        return DeiT(model_name=model_name, pretrained=pretrained, image_size=image_size)
    else:
        raise NotImplementedError(f"Requested {model_name} is not implemented.")


class Interpolation(nn.Module):
    """Interpolation nn.Module wrap for nn.functional.interpolate.

    Attributes:
        target_size (tuple[int, int] | torch.Size): target spatial size of this interpolation.
    """

    def __init__(self, target_size: tuple[int, int] | torch.Size) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Very simple forward pass to call interpolate()."""
        return interpolate(x, self.target_size)


class LinearAdapterHead(nn.Module):
    """Adapter head contains a single linear layer."""

    def __init__(
        self,
        source_size: tuple[int, ...] | torch.Size,
        target_size: tuple[int, ...] | torch.Size,
    ):
        """Initialization function for LinearAdapterHead.

        Args:
            source_size (tuple[int, ...] | torch.Size): the size of the source feature.
            target_size (tuple[int, ...] | torch.Size): the size of the target feature.
            num_layer (int): number of MLP layers (One linear layer if num_layer = 1).
        """
        super().__init__()

        self.source_size = source_size
        self.target_size = target_size

        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        self.adapter = nn.Sequential(
            nn.Linear(source_channel_size, target_channel_size),
        )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for the adapter."""
        assert not backbone_no_cls
        # x: [B, (1+H*W), C]
        # LinearAdapterHead is used only when there is cls token in the backbone.
        x = x[:, 0]
        x = self.adapter(x)
        return x  # [B, (H*W), C]


class MLPAdapterHead(nn.Module):
    """MLP Adapter module.

    Transforms features in shape source size [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Will first do interpolation to match the spatial size [H_t, W_t],
    followed by MLP to project to the target channel dimension [C_t].

    Attributes:
        source_size (tuple[int, ...] | torch.Size): the size of the source feature. [C, H, W]
        target_size (tuple[int, ...] | torch.Size): the size of the target feature. [C, H, W]
        adapter     (nn.Module):                    the adapter module.
        interpolation (nn.Module):                  interpolation to adjust sizes before MLP.
    """

    def __init__(
        self,
        source_size: tuple[int, ...] | torch.Size,
        target_size: tuple[int, ...] | torch.Size,
        num_layer: int,
    ):
        """Initialization function for MLPAdapter.

        Args:
            source_size (tuple[int, ...] | torch.Size): the size of the source feature.
            target_size (tuple[int, ...] | torch.Size): the size of the target feature.
            num_layer (int): number of MLP layers (One linear layer if num_layer = 1).
        """
        super().__init__()
        assert (
            num_layer >= 1
        ), f"`num_layer` in {self._get_name()} should >= 1. Got {num_layer}"

        self.source_size = source_size
        self.target_size = target_size

        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        self.interpolation = nn.Sequential(
            nn.Identity(),
        )
        if self.source_size[1] != self.target_size[1]:
            self.interpolation = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
                Interpolation(self.target_size[1:]),
                Rearrange("b c h w-> b (h w) c"),
            )

        if num_layer == 1:
            self.adapter = nn.Sequential(
                nn.Linear(source_channel_size, target_channel_size),
            )
        elif num_layer >= 2:
            hidden_dim = source_channel_size * 2
            self.adapter = nn.Sequential(
                nn.Linear(source_channel_size, hidden_dim),
                *list(
                    chain.from_iterable(
                        [
                            [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
                            for _ in range(num_layer - 2)
                        ]
                    )
                ),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_channel_size),
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for the adapter.

        First interpolation then MLP.
        """
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        # x: [B, (H*W), C]
        x = self.interpolation(x)
        x = self.adapter(x)
        return x  # [B, (H*W), C]


class ConvAdapterHead(nn.Module):
    """Convolutional Adapter module.

    Transforms features in shape source size [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Uses CNN to map channel and spatial sizes jointly.
    Note: only work for (16, 16), (any, any), any <= 14, and (64, 64) spatial sizes for now.

    Attributes:
        source_size (tuple[int, ...] | torch.Size): the size of the source feature.
        target_size (tuple[int, ...] | torch.Size): the size of the target feature.
        adapter     (nn.Module):                    the adapter module.
        interpolation (nn.Module):                  interpolation to adjust sizes before MLP.
    """

    def __init__(
        self,
        source_size: tuple[int, ...] | torch.Size,
        target_size: tuple[int, ...] | torch.Size,
    ):
        """Initialization function for ConvAdapter.

        Args:
            source_size (tuple[int, ...] | torch.Size): the size of the source feature.
            target_size (tuple[int, ...] | torch.Size): the size of the target feature.
        """
        super().__init__()
        self.source_size = source_size
        self.target_size = target_size

        hidden_dim = self.source_size[0] * 2
        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        if self.source_size[1] < 12:
            raise NotImplementedError(
                "feature spatial size smaller than 12x12 is not supported."
            )
        elif self.source_size[1] < 16:  # pad (any, any), any <= 14 to (16, 16)
            self.pad = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
                nn.ConvTranspose2d(
                    source_channel_size,
                    source_channel_size,
                    kernel_size=3,
                    stride=1,
                    output_padding=14 - self.source_size[1],
                ),
            )
            self.source_size = (self.source_size[0], 16, 16)
        elif (
            self.source_size[1] == 16 or self.source_size[1] == 64
        ):  # do nothing for (16, 16) and (64, 64)
            self.pad = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
            )
        else:
            raise NotImplementedError(
                "feature spatial size (>=16x16) other than 16x16 and 64x64 is not supported."
            )

        if self.source_size[1] < self.target_size[1]:  # (16, 16) / (14, 14) to (64, 64)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.ConvTranspose2d(
                    source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 31
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 31, 31]),
                nn.ConvTranspose2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, output_padding=1
                ),  # 64
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 64, 64]),
                nn.ConvTranspose2d(
                    hidden_dim, target_channel_size, kernel_size=3, stride=1, padding=1
                ),  # 64
                Rearrange("b c h w-> b (h w) c"),
            )
        elif self.source_size[1] == self.target_size[1]:  # (16, 16) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(
                    source_channel_size, hidden_dim, kernel_size=3, padding=1
                ),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(
                    hidden_dim, target_channel_size, kernel_size=3, padding=1
                ),  # 16
                Rearrange("b c h w-> b (h w) c"),
            )
        else:  # (64, 64) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(
                    source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 32, 32]),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 16]),
                nn.Conv2d(
                    hidden_dim, target_channel_size, kernel_size=3, padding=1
                ),  # 16
                Rearrange("b c h w-> b (h w) c"),
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for ConvAdapter."""
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        # x: [B, H*W, C]
        x = self.pad(x)
        x = self.adapter(x)
        return x  # B, (H*W), C


class LightConvAdapterHead(nn.Module):
    """Light Convolutional Adapter module.

    Transforms features from source size in [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Uses CNN to map channel and spatial sizes jointly.
    Note: only work for source sizes (H_s, W_s): (16, 16), (any, any), 12 <= any <= 14,
        and target sizes (H_t, W_t): (16, 16) and (64, 64) for now.

    Attributes:
        source_size (tuple[int, ...] | torch.Size): the size of the source feature,
            channel first (C, H, W).
        target_size (tuple[int, ...] | torch.Size): the size of the target feature,
            channel first (C, H, W).
        adapter     (nn.Module):                    the adapter module.
        interpolation (nn.Module):                  interpolation to adjust sizes before MLP.
    """

    def __init__(
        self,
        source_size: tuple[int, ...] | torch.Size,
        target_size: tuple[int, ...] | torch.Size,
        hidden_size_factor: int | float = 1.0,
    ):
        """Initialization function for ConvAdapter.

        Args:
            source_size (tuple[int, ...] | torch.Size): the size of the source feature.
            target_size (tuple[int, ...] | torch.Size): the size of the target feature.
            hidden_size_factor (int | float): the size of hidden dim of feature translator
                as a factor of input feature hidden dim.
        """
        super().__init__()
        if source_size[1] != source_size[2] or target_size[1] != target_size[2]:
            raise NotImplementedError(
                "Currently does not support non-square feature maps like source size"
                "{source_size} and target size {target_size}."
            )
        self.source_size = source_size
        self.target_size = target_size
        self.hidden_size_factor = hidden_size_factor

        hidden_dim = int(self.source_size[0] * hidden_size_factor)
        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        if self.source_size[1] < 12:
            raise NotImplementedError(
                "feature spatial size smaller than 12x12 is not supported."
            )
        elif (
            self.source_size[1] < 16 and self.target_size[1] >= 16
        ):  # pad (any, any), any <= 14 to (16, 16)
            self.pad = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
                nn.ConvTranspose2d(
                    source_channel_size,
                    source_channel_size,
                    kernel_size=3,
                    stride=1,
                    output_padding=14 - self.source_size[1],
                ),
            )
            self.source_size = (self.source_size[0], 16, 16)
        elif (self.source_size[1] == 16 or self.source_size[1] == 64) or (
            self.source_size[1] == 14 and self.target_size[1] == 14
        ):
            # no padding for (16, 16), (64, 64) and (14, 14) <-> (14, 14)
            self.pad = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
            )
        elif self.target_size[1] < 14:
            self.pad = nn.Sequential(
                Rearrange(
                    "b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]
                ),
            )
        else:
            raise NotImplementedError(
                "feature spatial size larger than 16x16 (other than 64x64) is not supported."
            )

        if (
            self.source_size[1] == 16 and self.target_size[1] == 64
        ):  # (16, 16) to (64, 64)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.ConvTranspose2d(
                    source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 31
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 31, 31]),
                nn.ConvTranspose2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, output_padding=1
                ),  # 64
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 64, 64]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        elif self.source_size[1] == self.target_size[1]:  # (16, 16) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(
                    source_channel_size, hidden_dim, kernel_size=3, padding=1
                ),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        elif (
            self.source_size[1] == 64 and self.target_size[1] == 16
        ):  # (64, 64) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(
                    source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 32, 32]),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 16]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        elif self.target_size[1] == 7:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(
                    source_channel_size, hidden_dim, kernel_size=4, stride=2, padding=1
                ),  # 14x14 -> 7x7
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 7, 7]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        else:
            NotImplementedError(
                f"{self.source_size} to {self.target_size} is not supported."
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for ConvAdapter."""
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        x = self.pad(x)
        x = self.adapter(x)
        return x  # [B, H*W, C]


class FeatureTranslator(nn.Module):
    """Base class for the feature translator.

    The flow is backbone_adapter -> translator_stem -> translator_heads.

    Attributes:
        backbone_feature_size (torch.Size): the size of features of the backbone.
        target_feature_sizes (dict[str, torch.Size | tuple[int, ...]]): the sizes of features of target models.
        translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        target_model_names (list[str]): convenient attribute to hold all the names of the target models.

        backbone_adapter (nn.Module): the adapter to map channel dim of backbone to the translator hidden dim.
        translator_stem (nn.Module):  the shared stem for all target models.
        translator_heads (nn.ModuleDict): specific heads for different target models.
    """

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
    ) -> None:
        """Initialization function for FeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size | tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        """
        super().__init__()
        self.backbone_feature_size = backbone_feature_size  # (C, H, W)
        self.target_feature_sizes = target_feature_sizes  # [(C, H, W)]
        self.translator_hidden_size = translator_hidden_size  # C
        self.target_model_names = list(target_feature_sizes.keys())
        self.legit_target_model_name_map: dict[str, str] = {
            t: t.replace(".", "_") for t in self.target_model_names
        }
        self.translator_heads: nn.ModuleDict = None

        self.backbone_adapter = nn.Sequential(
            nn.LayerNorm(self.backbone_feature_size[0]),  # do a pre-norm
            nn.Linear(
                self.backbone_feature_size[0],  # C in [C,H,W]
                self.translator_hidden_size,
            ),
        )
        self.translator_stem: nn.Module = nn.Identity()
        self.build_translator_heads()

    def build_translator_heads(self) -> None:
        """Build translator heads to match the dimension of each target feature set.

        Example:
            translator_heads: dict[str, nn.Module] = ...
            self.translator_heads = nn.ModuleDict(translator_heads)
        """
        raise NotImplementedError("build_translator_heads() should be overridden")

    def forward(
        self,
        x: torch.Tensor,
        target_model_names: list[str] | None = None,
        backbone_no_cls: bool = False,
    ) -> torch.Tensor:
        """Forward pass for a base feature translator.

        Args:
            x (torch.Tensor): input features from the backbone. [B, (1)+H*W, C].
                (1) means optional CLS token. If `backbone_no_cls==True`, then [B, H*W, C].
            target_model_names (Optional[list[str]]): names of the target models.
            backbone_no_cls (bool): indicate backbone has cls token or not.
                Can use it to customize whether to drop cls.

        Returns:
            dict[str, torch.Tensor]: predicted features for target models.
        """
        # x: [B, (1)+H*W, C]
        x = self.backbone_adapter(x)
        x = self.translator_stem(x)
        target_model_names = (
            target_model_names
            if target_model_names is not None
            else self.target_model_names
        )
        features = {
            t: self.translator_heads[self.legit_target_model_name_map[t]](
                x, backbone_no_cls=backbone_no_cls
            )
            for t in target_model_names
        }
        return features


class MLPFeatureTranslator(FeatureTranslator):

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
        translator_n_layer: int = 3,
    ) -> None:
        """Initialization function for MLPFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 2048.
            translator_n_layer (int): number of MLP layers. Defaults to 3.
        """
        self.translator_n_layer = translator_n_layer

        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build MLP translator heads to match the dimension of each target feature set."""
        translator_heads = {}
        source_size = (self.translator_hidden_size, *self.backbone_feature_size[1:])
        for target_model, target_size in self.target_feature_sizes.items():
            head = MLPAdapterHead(
                source_size=source_size,
                target_size=target_size,
                num_layer=self.translator_n_layer,
            )
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class ConvFeatureTranslator(FeatureTranslator):

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
    ) -> None:
        """Initialization function for ConvFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 2048.
        """
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build translator heads to match the dimension of each target feature set.

        Returns:
            nn.ModuleDict: the translator heads.
        """
        translator_heads = {}
        source_size = (self.translator_hidden_size, *self.backbone_feature_size[1:])
        for target_model, target_size in self.target_feature_sizes.items():
            head = ConvAdapterHead(source_size=source_size, target_size=target_size)
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class LightConvFeatureTranslator(FeatureTranslator):

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]],
        translator_hidden_size: int = 1024,
        hidden_size_factor: int | float = 1.0,
    ) -> None:
        """Initialization function for LightConvFeatureTranslator.
            It's for a smaller translator compared to ConvFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (dict[str, torch.Size  |  tuple[int, ...]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 1024.
            hidden_size_factor: the size of hidden dim of feature translator
                as a factor of input feature hidden dim. Defaults to 1.0
        """
        self.hidden_size_factor = hidden_size_factor
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )
        self.backbone_adapter = nn.Identity()

    def build_translator_heads(self) -> nn.ModuleDict:
        """Build translator heads to match the dimension of each target feature set.

        Returns:
            nn.ModuleDict: the translator heads.
        """
        translator_heads = {}
        for target_model, target_size in self.target_feature_sizes.items():
            if "_cls" in target_model:
                head = LinearAdapterHead(
                    source_size=self.backbone_feature_size, target_size=target_size
                )
            else:
                head = LightConvAdapterHead(
                    source_size=self.backbone_feature_size,
                    target_size=target_size,
                    hidden_size_factor=self.hidden_size_factor,
                )
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


class TransformerFreatureTranslator(FeatureTranslator):

    def __init__(
        self,
        backbone_feature_size: torch.Size,
        target_feature_sizes: dict[str, torch.Size | tuple[int, int]],
        translator_hidden_size: int = 1024,
        translator_n_layers: int = 2,
        translator_n_heads: int = 8,
        translator_activation: str = "gelu",
    ) -> None:
        super().__init__(
            backbone_feature_size=backbone_feature_size,
            target_feature_sizes=target_feature_sizes,
            translator_hidden_size=translator_hidden_size,
        )

        self.translator_stem = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=translator_hidden_size,
                nhead=translator_n_heads,
                dim_feedforward=translator_hidden_size * 2,
                activation=translator_activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=translator_n_layers,
        )

        self.decode_tokens = nn.Parameter(
            torch.randn(
                (1, math.prod(self.backbone_feature_size[1:]), translator_hidden_size)
            )
        )

        self.target_model_emb = nn.ParameterDict(
            {
                self.legit_target_model_name_map[t]: torch.randn(
                    1, 1, translator_hidden_size
                )
                for t in self.target_model_names
            }
        )

    def build_translator_heads(self) -> None:
        """Build Transformer translator heads to match the dimension of each target feature set."""
        translator_heads = {}
        for target_model, target_size in self.target_feature_sizes.items():
            head = MLPAdapterHead(
                source_size=(
                    self.translator_hidden_size,
                    *self.backbone_feature_size[1:],
                ),
                target_size=target_size,
                num_layer=2,
            )
            translator_heads[self.legit_target_model_name_map[target_model]] = head
        self.translator_heads = nn.ModuleDict(translator_heads)

    def forward(
        self,
        x: torch.Tensor,
        target_model_names: list[str] | None = None,
        backbone_no_cls: bool = False,
    ) -> torch.Tensor:
        """Forward pass for a simple linear translator.

        Args:
            x (torch.Tensor): input features from the backbone.
            target_model_names (Optional[str]): names of the target models.
            backbone_no_cls (bool): indicate backbone has cls token or not.
                Can use it to customize whether to drop cls.

        Returns:
            dict[str, torch.Tensor]: predicted features for target models.
        """
        if not backbone_no_cls:
            x = x[:, 1:]
        x = self.backbone_adapter(x)
        features = {}
        target_model_names = (
            target_model_names
            if target_model_names is not None
            else self.target_model_names
        )
        for t in target_model_names:
            feature = self.translator_stem(
                torch.cat(
                    [
                        self.decode_tokens.repeat(x.size(0), 1, 1),
                        self.target_model_emb[
                            self.legit_target_model_name_map[t]
                        ].repeat(x.size(0), 1, 1),
                    ],
                    dim=1,
                ),
                memory=x,
            )[:, 1:, ...]
            features[t] = self.translator_heads[self.legit_target_model_name_map[t]](
                feature
            )
        return features


def build_feature_translator(translator_type: str, **kwargs: Any) -> FeatureTranslator:
    """Handy function to build feature translators given the type.

    Args:
        translator_type (str): the type of the translator,
            one in `"mlp"`, `"conv"`, `"lconv"`, `"transformer"` (or `"trans"`).
            At the moment we are actively using `"lconv"`.

    Returns:
        FeatureTranslator: the corresponding FeatureTranslator
    """
    if translator_type == "mlp":
        return MLPFeatureTranslator(**kwargs)
    elif translator_type == "conv":
        return ConvFeatureTranslator(**kwargs)
    elif translator_type == "lconv":
        return LightConvFeatureTranslator(**kwargs)
    elif translator_type == "transformer" or translator_type == "trans":
        return TransformerFreatureTranslator(**kwargs)
    else:
        raise NotImplementedError(
            f"Requested {translator_type} is not implemented yet."
        )


class TheiaConfig(PretrainedConfig):

    def __init__(
        self,
        backbone: str | nn.Module = "facebook/deit-tiny-patch16-224",
        pretrained: bool = False,
        target_feature_sizes: dict[str, torch.Size | tuple[int, ...]] | None = None,
        translator_type: str = "lconv",
        translator_hidden_size_factor: float | int = 1.0,
        target_loss_weights: dict[str, float] | None = None,
        feature_reduce_method: str | None = None,
        feature_neck: bool = False,
        feature_neck_hidden_dim: int = 256,
        forward_neck: bool = False,
        feature_neck_nonlinearity: str = "relu",
        iamge_size: int = 224,
        num_reg_tokens: int = 0,
        **kwargs: Any,
    ):
        self.backbone = backbone
        self.pretrained = pretrained
        self.target_feature_sizes = target_feature_sizes
        self.translator_type = translator_type
        self.translator_hidden_size_factor = translator_hidden_size_factor
        self.target_loss_weights = target_loss_weights
        self.feature_reduce_method = feature_reduce_method
        self.feature_neck = feature_neck
        self.feature_neck_hidden_dim = feature_neck_hidden_dim
        self.forward_neck = forward_neck
        self.feature_neck_nonlinearity = feature_neck_nonlinearity
        self.image_size = 224
        self.num_reg_tokens = num_reg_tokens
        super().__init__(**kwargs)


class TheiaModel(PreTrainedModel):
    config_class = TheiaConfig

    def __init__(self, config: TheiaConfig):
        super().__init__(config)

        self.target_feature_sizes = config.target_feature_sizes
        self.preprocessor = None
        self.pretrained = config.pretrained

        # backbone
        self.image_size = config.image_size
        if "reg" in config.backbone:
            self.backbone: nn.Module = build_backbone(
                config.backbone,
                config.pretrained,
                image_size=config.image_size,
                num_reg_tokens=config.num_reg_tokens,
            )
        else:
            self.backbone: nn.Module = build_backbone(
                config.backbone, config.pretrained, image_size=config.image_size
            )

        # handle output feature (feature reduce)
        self.feature_reduce_method = config.feature_reduce_method
        self.no_cls = hasattr(self.backbone, "no_cls")
        self.num_reg_tokens = (
            self.backbone.num_reg_tokens
            if hasattr(self.backbone, "num_reg_tokens")
            else 0
        )

        # translator
        backbone_feature_size = self.backbone.get_feature_size(keep_spatial=True)
        if self.target_feature_sizes:
            translator_kwargs = {
                "hidden_size_factor": config.translator_hidden_size_factor
            }
            translator_kwargs["backbone_feature_size"] = backbone_feature_size
            translator_kwargs["target_feature_sizes"] = config.target_feature_sizes
            self.translator = build_feature_translator(
                config.translator_type, **translator_kwargs
            )
        else:
            self.translator = None

        self.feature_neck = config.feature_neck
        self.feature_neck_hidden_dim = config.feature_neck_hidden_dim
        self.forward_neck = config.forward_neck
        if self.feature_neck:
            num_tokens_edge = (
                self.backbone.model.config.image_size
                // self.backbone.model.config.patch_size
            )
            self.neck = nn.Sequential(
                Rearrange("b (h w) c -> b c h w", h=num_tokens_edge, w=num_tokens_edge),
                nn.Conv2d(
                    self.backbone.model.config.hidden_size,
                    self.feature_neck_hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),  # 14x14 -> 7x7
                (
                    nn.ReLU()
                    if config.feature_neck_nonlinearity == "relu"
                    else nn.Tanh()
                ),  # just to keep the same as super class
                nn.Conv2d(
                    self.feature_neck_hidden_dim,
                    self.feature_neck_hidden_dim,
                    kernel_size=3,
                    stride=2,
                ),  # 7x7 -> 3x3
                nn.ReLU() if config.feature_neck_nonlinearity == "relu" else nn.Tanh(),
                nn.Conv2d(
                    self.feature_neck_hidden_dim,
                    self.feature_neck_hidden_dim,
                    kernel_size=3,
                    stride=1,
                ),  # 3x3 -> 1x1
                nn.ReLU() if config.feature_neck_nonlinearity == "relu" else nn.Tanh(),
                nn.Flatten(),
            )
        else:
            self.neck = None

        # loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.cos_target = torch.ones((1), dtype=torch.int, requires_grad=False)
        self.target_loss_weights = config.target_loss_weights

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load weights from `checkpoint_path` manually.

        Args:
            checkpoint_path (str): path to the weights.
        """
        # load theia weights
        if checkpoint_path:
            weights_dict = torch.load(checkpoint_path, map_location="cpu")
            # Filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in weights_dict.items() if k in self.state_dict()
            }
            self.load_state_dict(pretrained_dict, strict=False)

    def freeze_translator(self) -> None:
        """Freeze feature translators `self.translator`."""
        if self.translator is not None:
            for param in self.translator.parameters():
                param.requires_grad = False

    def freeze_backbone(self) -> None:
        """Freeze backbone (encoder) `self.backbone`."""
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze backbone (encoder) `self.backbone`."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_neck(self) -> None:
        """Freeze feature neck `self.neck`."""
        if self.neck is not None:
            for param in self.neck.parameters():
                param.requires_grad = False

    def freeze_everything(self) -> None:
        """Freeze all parameters in the model."""
        self.freeze_translator()
        self.freeze_neck()
        self.freeze_encoder()

    def unfreeze_translator(self) -> None:
        if self.translator is not None:
            for param in self.translator.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        "Set parameters in backbone (encoder) `self.backbone` trainable."
        self.unfreeze_encoder()

    def unfreeze_encoder(self) -> None:
        "Set parameters in backbone (encoder) `self.backbone` trainable."
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_neck(self) -> None:
        "Set parameters in feature neck `self.neck` trainable."
        if self.neck is not None:
            for param in self.neck.parameters():
                param.requires_grad = True

    def unfreeze_everything(self) -> None:
        """Set all parameters trainable."""
        self.unfreeze_translator()
        self.unfreeze_neck()
        self.unfreeze_encoder()

    def set_forward_neck(self, forward_neck: bool = True) -> None:
        """Set `self.forward_neck` to `forward_neck` value.

        Args:
            forward_neck (bool): whether forward the feature through the random initialized neck.
                If set to True, the output from `self.forward()` will be in shape [batch_size, self.config.feature_neck_hidden_dim]
        """
        self.forward_neck = forward_neck

    def forward_feature(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward RVFM feature only (before translators).

        Args:
            x (torch.Tensor): input image. By default it accepts images
                in shape [B, H, W, C] or [B, C, H, W], pixel range [0,255], torch.uint8.
            kwargs (Any): kwargs including mainly those for huggingface preprocessor:
                `do_resize` (bool) defaults to True.
                `interpolate_pos_encoding` (Optional[bool]) defaults to None.
                `do_rescale` (bool) defaults to True.
                `do_normalize` (bool) defaults to True.

        Returns:
            torch.Tensor: RVFM feature.
        """
        feature = self.backbone(x, **kwargs)
        # [B, 1+H*W+N, C] if including both CLS and register tokens.
        # [B, 1+H*W, C] for standard model (N=0).
        # [B, H*W, C] for model without CLS.
        return handle_feature_output(feature, num_discard_tokens=self.num_reg_tokens)

    def forward(
        self,
        x: torch.Tensor,
        target_model_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass of Robot Vision Foundation Model.

        Args:
             x (torch.Tensor): input image. By default it accepts images
                 in shape [B, H, W, C] or [B, C, H, W], pixel range [0,255], torch.uint8.
             target_model_names (Optional[list[str]]): names of the target foundation models.
             kwargs (Any): kwargs including mainly those for huggingface preprocessor:
                 `do_resize` (bool) defaults to True.
                 `interpolate_pos_encoding` (Optional[bool]) defaults to None.
                 `do_rescale` (bool) defaults to True.
                 `do_normalize` (bool) defaults to True.

         Returns:
         if `self.forward_neck`:
             torch.Tensor: compact vector feature passed through the neck. [B, C_neck]
         else:
             dict[str, torch.Tensor]: features that match to each foundation model.
                 Each feature is in [B, (H*W), C] or [B, C].
        """
        if self.forward_neck:
            x = self.forward_feature(x)
            return self.neck(x)
        else:
            x = self.backbone(x, **kwargs)
            if self.num_reg_tokens > 0:
                x = x[:, : -self.num_reg_tokens]  # [B, (1)+H*W, C]
            features = self.translator(
                x, target_model_names, backbone_no_cls=self.no_cls
            )  # each is [B, H*W, C] or [B, C]
            return features

    def get_loss(
        self, pred_features: dict[str, torch.Tensor], y: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Get loss terms given predictions and targets.

        Args:
            pred_features (dict[str, torch.Tensor]): predictions.
            y (dict[str, torch.Tensor]): targets.

        Returns:
            tuple[Any, ...]: loss terms
        """
        mse_loss_avg, cos_loss_avg, l1_loss_avg = 0, 0, 0
        mse_losses_per_model = {}
        cos_losses_per_model = {}
        l1_losses_per_model = {}

        for t in pred_features:
            pred = pred_features[t]
            target = y[t]

            # mse loss
            mse_loss = self.mse_loss(pred, target)
            weight = (
                self.target_loss_weights
                if self.target_loss_weights
                else 1.0 / len(pred_features)
            )

            # l1 loss
            l1_loss = self.l1_loss(pred, target)

            # cos loss
            pred_norm = F.normalize(pred.flatten(start_dim=1), dim=1, p=2)
            target_norm = F.normalize(target.flatten(start_dim=1), dim=1, p=2)
            target = self.cos_target.repeat(pred.size(0)).to(pred.device)
            cos_loss = self.cos_loss(pred_norm, target_norm, target)

            mse_loss_avg += mse_loss * weight
            cos_loss_avg += cos_loss / len(
                pred_features
            )  # balance cos by default for meaningful eval
            l1_loss_avg += l1_loss * weight

            mse_losses_per_model[t] = mse_loss.item()
            cos_losses_per_model[t] = cos_loss.item()
            l1_losses_per_model[t] = l1_loss.item()

        return {
            "mse_loss": mse_loss_avg,
            "cos_loss": cos_loss_avg,
            "l1_loss": l1_loss_avg,
            "mse_losses_per_model": mse_losses_per_model,
            "cos_losses_per_model": cos_losses_per_model,
            "l1_losses_per_model": l1_losses_per_model,
        }
