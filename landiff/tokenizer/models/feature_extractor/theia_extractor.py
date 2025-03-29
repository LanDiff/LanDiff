import sys

import einops
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torch import Tensor
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig

from landiff.utils import freeze_model

from .theia_model import TheiaModel


class TheiaExtractor(nn.Module):

    def __init__(
        self,
        *,
        pretrained_model_name_or_path="theaiinstitute/theia-base-patch16-224-cddsv",
        image_size=(224, 224),
        micro_batch_size=sys.maxsize,
        interpolate=False,
        output_shape=None,
        bfp16=False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        self.model = TheiaModel.from_pretrained(
            pretrained_model_name_or_path, config=config, trust_remote_code=True
        )
        self.bfp16 = bfp16
        self.micro_batch_size = micro_batch_size
        self.image_size = image_size
        self.interpolate = interpolate
        self.output_shape = output_shape
        freeze_model(self)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, images: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass of the feature extractor model.

        This method processes input images through a feature extraction model, handling batching
        and optional image resizing. It expects images in uint8 format with values in [0, 255].

        Args:
            images (Tensor): Input images tensor with shape (..., C, H, W) in uint8 format.
                            The tensor should have at least 4 dimensions.

        Returns:
            Tensor: Extracted features with shape (*prefix_shape, C, h, w), where:
                   - prefix_shape matches the input batch dimensions
                   - C is the feature channel dimension
                   - h, w are spatial dimensions, possibly resized based on output_shape

        Note:
            - Input images are processed in micro-batches for memory efficiency
            - If self.interpolate is True, uses model's internal resizing with interpolated
              position encoding
            - If self.interpolate is False, images are resized to self.image_size using
              Lanczos interpolation
            - Output features can be resized/padded to match self.output_shape if specified
            - When self.bfp16 is True, computation is done in bfloat16 precision
        """
        # 约定传入的images shape 为 (..., C, H, W)，格式为uint8,[0,255]
        # 检查是否符合约定
        assert (
            images.ndim >= 4
        ), f"images should have 4 dimensions, but got {images.ndim}"
        prefix_shape = images.shape[:-3]
        images = einops.rearrange(images, "... c h w -> (...) c h w")
        assert (
            images.dtype == torch.uint8
        ), f"images should have dtype uint8, but got {images.dtype}"
        assert (
            images.min() >= 0
        ), f"images should have min value >= 0, but got {images.min()}"
        assert (
            images.max() <= 255
        ), f"images should have max value <= 255, but got {images.max()}"
        if not self.interpolate:
            images = v2.functional.resize(
                images, self.image_size, interpolation=InterpolationMode.LANCZOS
            )
        images = einops.rearrange(images, "n c h w -> n h w c")
        features_list = []
        with (
            torch.autocast(
                enabled=True, device_type=images.device.type, dtype=torch.bfloat16
            )
            if self.bfp16
            else torch.autocast(enabled=False, device_type=images.device.type)
        ):
            for i in range(0, len(images), self.micro_batch_size):
                batch_images = images[i : i + self.micro_batch_size].clone()
                if self.interpolate:
                    batch_features: Tensor = self.model.forward_feature(
                        batch_images, do_resize=False, interpolate_pos_encoding=True
                    )  # shape (n,l,c)
                else:
                    batch_features: Tensor = self.model.forward_feature(
                        batch_images
                    )  # shape (n,l,c)
                h_w = batch_features.shape[1]
                h_w = int(h_w**0.5)
                assert (
                    h_w * h_w == batch_features.shape[1]
                ), f"l should be a square number, but got {batch_features.shape[1]}"
                batch_features = einops.rearrange(
                    batch_features, "n (h w) c -> n c h w", h=h_w
                )
                if self.output_shape is not None:
                    if (
                        self.output_shape[0] < batch_features.shape[-1]
                        and self.output_shape[1] < batch_features.shape[-2]
                    ):
                        batch_features = batch_features[
                            ..., : self.output_shape[0], : self.output_shape[1]
                        ]
                    else:
                        # pad
                        pad = (
                            self.output_shape[1] - batch_features.shape[-2],
                            self.output_shape[0] - batch_features.shape[-1],
                        )
                        pad = [max(i, 0) for i in pad]
                        batch_features = torch.nn.functional.pad(
                            batch_features, (0, pad[0], 0, pad[1])
                        )
                        batch_features = batch_features[
                            ..., : self.output_shape[0], : self.output_shape[1]
                        ]
                features_list.append(batch_features.clone())
        features = torch.cat(features_list, dim=0)  # shape (n,c,h,w)
        features = features.reshape(*prefix_shape, *features.shape[-3:])
        return features
