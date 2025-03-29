#!/usr/bin/env python
import logging

import einops
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

from landiff.diffusion.sgm.util import instantiate_from_config
from landiff.utils import zero_module

logger = logging.getLogger(__name__)


def pad_to_square(input, pad_value):
    # shape ...,C,H,W
    h, w = input.shape[-2:]
    if h == w:
        return input
    # pad pattern left top right bottom,注意 torchvision pad的顺序和torch.nn.functional.pad不一样
    if h > w:
        pad = (0, 0, h - w, 0)
    else:
        pad = (0, 0, 0, w - h)
    input = v2.functional.pad(input, pad, fill=pad_value)
    assert input.shape[-2] == input.shape[-1]
    return input


class SemanticCond(nn.Module):

    def __init__(
        self,
        *,
        semantic_model_config: nn.Module,
        upsample_model_config: nn.Module,
        dtype,
        out_dim,
        target_dim,
        dowsample_factor=16,
        feature_type: str = "video_theia_interpolate",
        zero_init_conv_out: bool = True,
        augmenter_params: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.semantic_model = instantiate_from_config(semantic_model_config)
        if upsample_model_config is not None:
            self.upsample_model = instantiate_from_config(upsample_model_config)
        if zero_init_conv_out:
            self.conv_out = zero_module(
                torch.nn.Conv2d(out_dim, target_dim, kernel_size=3, stride=1, padding=1)
            )  # zero 初始化
        else:
            self.conv_out = torch.nn.Conv2d(
                out_dim, target_dim, kernel_size=3, stride=1, padding=1
            )

        self.feature_type = feature_type
        self.dowsample_factor = dowsample_factor
        self.dtype = dtype
        if self.feature_type == "video_theia_interpolate":
            self.pad_values = [127, 127, 127]
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    @property
    def device(self):
        return next(self.upsample_model.parameters()).device

    def theia_forward(self, visual: torch.Tensor) -> torch.Tensor:
        visual = einops.rearrange(visual, "B T C H W -> (B T) C H W")
        features = self.semantic_model(visual)
        features = torch.nn.functional.interpolate(
            features, self.input_shape, mode="bilinear"
        )
        if features.dtype != self.dtype:
            features = features.to(self.dtype)

        features = self.upsample_model(features)
        features = torch.nn.functional.interpolate(
            features, self.output_shape, mode="bilinear"
        )
        return features

    def video_theia_interpolate_forward(
        self,
        visual: torch.Tensor = None,
        indexs: torch.Tensor = None,
        semantic_feature_before_upsample: torch.Tensor = None,
    ) -> torch.Tensor:
        if visual is not None:
            origin_h, origin_w = visual.shape[-2:]
            target_h = origin_h // self.dowsample_factor
            target_w = origin_w // self.dowsample_factor
            visual = pad_to_square(visual, self.pad_values)
            B, T = visual.shape[:2]
        if semantic_feature_before_upsample is None:
            features = self.semantic_model(visual, indexs)
        else:
            features = semantic_feature_before_upsample
        if visual is not None:
            features = features[..., :target_h, :target_w]
        if features.dtype != self.dtype:
            features = features.to(self.dtype)
        B, T = features.shape[:2]
        features = einops.rearrange(features, "B T C H W -> (B T) C H W")
        features = self.upsample_model(features)
        features = einops.rearrange(features, "(B T) C H W -> B T C H W", B=B, T=T)
        return features

    def forward(
        self,
        visual: torch.Tensor = None,
        indexs: torch.Tensor = None,
        vq_origin_features: torch.Tensor = None,
        semantic_feature_before_upsample: torch.Tensor = None,
    ) -> torch.Tensor:
        # shape: (B, T,C,H,W),[-1,1]
        # [-1,1] to [0,1]
        if visual is not None:
            visual = (visual + 1.0) / 2.0
            visual = visual.clamp(0, 1)
            visual = v2.functional.to_dtype(visual, dtype=torch.uint8, scale=True)
        if self.feature_type == "video_theia_interpolate":
            features = self.video_theia_interpolate_forward(
                visual,
                indexs,
                semantic_feature_before_upsample=semantic_feature_before_upsample,
            )
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        B, T = features.shape[:2]
        features = einops.rearrange(features, "B T C H W -> (B T) C H W")
        features = self.conv_out(features)
        features = einops.rearrange(features, "(B T) C H W -> B T C H W", B=B, T=T)
        return features
