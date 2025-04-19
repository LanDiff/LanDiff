import importlib
import os
from contextlib import nullcontext

import fiddle as fdl
import torch
import torch.nn as nn
from fiddle import Config
from safetensors.torch import load_file
from torch import Tensor

from landiff.diffusion.sgm.modules.diffusionmodules.loss import ExtraLossRegistry
from landiff.tokenizer.models.video_titok_vq import VideoVQ
from landiff.utils import freeze_model as freeze_model_fn


class VQWarp(nn.Module):
    def __init__(
        self,
        config_str: str,
        ckpt_path: str = None,
        freeze_model=True,
        freeze_encoder=False,
    ):
        super(VQWarp, self).__init__()

        self.freeze_model = freeze_model
        self.freeze_encoder = freeze_encoder
        # 动态导入配置函数
        module_path, function_name = config_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_function = getattr(module, function_name)

        # 调用配置函数获取配置
        model_config: Config = config_function()

        self.model: VideoVQ = fdl.build(model_config)
        if ckpt_path is not None:
            assert os.path.exists(
                ckpt_path
            ), f"Please provide the correct path of the weight of the tokenizer, the path you provide is: {ckpt_path}"
            assert ckpt_path.endswith(
                "safetensors"
            ), f"Must be safetensors file, the path you provide is: {ckpt_path}"
            if ckpt_path.endswith("safetensors"):
                state_dict = load_file(ckpt_path)
                self.model.load_state_dict(state_dict, strict=True)
                del state_dict

        assert not (
            self.freeze_encoder and self.freeze_model
        ), "Can't freeze both encoder and model"
        if freeze_model:
            freeze_model_fn(self.model, disable_state_dict=False)
        if freeze_encoder:
            freeze_model_fn(self.model.encoder, disable_state_dict=False)
            freeze_model_fn(self.model.quantizer, disable_state_dict=False)

    def forward(self, images: Tensor) -> Tensor:
        # 约定传入的images shape 为 (N, C, H, W)，格式为uint8,[0,255]
        # 检查是否符合约定
        assert (
            images.ndim == 4
        ), f"images should have 4 dimensions, but got {images.ndim}"
        assert (
            images.dtype == torch.uint8
        ), f"images should have dtype uint8, but got {images.dtype}"
        assert (
            images.min() >= 0
        ), f"images should have min value >= 0, but got {images.min()}"
        assert (
            images.max() <= 255
        ), f"images should have max value <= 255, but got {images.max()}"

        with torch.no_grad() if self.freeze_model else nullcontext():
            result = self.model(images, return_features=True)

        re_features = result["re_features"]
        commit_loss = result["commit_loss"]
        re_loss = result["re_loss"]
        ExtraLossRegistry.register("commit_loss", commit_loss)
        ExtraLossRegistry.register("re_loss", re_loss)
        re_features = self.model.denormalize(re_features)
        return re_features


class VideoVQWrap(VQWarp):
    def forward(
        self, images: Tensor = None, indexs: Tensor = None, features: Tensor = None
    ) -> Tensor:
        if not self.training and indexs is not None:
            return self.model.index_to_feature(indexs)
        if not self.training and features is not None:
            return features
        # 约定传入的images shape 为 (N, T, C, H, W)，格式为uint8,[0,255]
        # 检查是否符合约定
        assert (
            images.ndim == 5
        ), f"images should have 5 dimensions, but got {images.ndim}"
        assert (
            images.dtype == torch.uint8
        ), f"images should have dtype uint8, but got {images.dtype}"
        assert (
            images.min() >= 0
        ), f"images should have min value >= 0, but got {images.min()}"
        assert (
            images.max() <= 255
        ), f"images should have max value <= 255, but got {images.max()}"

        with torch.no_grad() if self.freeze_model else nullcontext():
            result = self.model(
                {
                    "video": images,
                    "features": features,
                },
                return_features=True,
            )

        re_features = result["re_features"]
        commit_loss = result["commit_loss"]
        re_loss = result["re_loss"]
        ExtraLossRegistry.register("commit_loss", commit_loss)
        ExtraLossRegistry.register("re_loss", re_loss)
        re_features = self.model.denormalize(re_features)
        return re_features
