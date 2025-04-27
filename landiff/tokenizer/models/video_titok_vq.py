import os
from contextlib import nullcontext
from pathlib import Path

import einops
import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor
from vector_quantize_pytorch import FSQ, VectorQuantize

from landiff.utils import freeze_model, maybe_autocast


class TowDVQ(nn.Module):

    def __init__(
        self,
        *,
        feature_extractor,
        encoder,
        decoder,
        quantizer: FSQ | VectorQuantize,
        re_loss_fn,
        online_mean_std=None,
        mean_std_path: str | Path | None = None,
        mean_std_dim: int | None = None,
        commit_loss_weight=1.0,
        recon_loss_weight=1.0,
        num_latent_tokens=None,
        fwd_dtype=torch.float32,
        model_type="cnn",  # cnn transformer var_len_transformer
        ckpt_path: None | str = None,
        freeze=False,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.online_mean_std = online_mean_std
        self.re_loss_fn = re_loss_fn
        self.commit_loss_weight = commit_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.fwd_dtype = fwd_dtype
        self.ckpt_path = ckpt_path
        self.freeze = freeze

        self.mean_std_path = mean_std_path
        if self.online_mean_std is not None:
            self.dump_emb = nn.Parameter(torch.rand(2))
        # Initialize buffers with default values
        if mean_std_dim is not None:
            self.register_buffer("mean", torch.zeros(mean_std_dim))
            self.register_buffer("std", torch.ones(mean_std_dim))

        # Override with values from file if available
        if self.mean_std_path is not None:
            mean_std_path = Path(mean_std_path)
            if mean_std_path.exists():
                mean_std = torch.load(mean_std_path)
                if hasattr(self, "mean"):
                    self.mean.copy_(mean_std[0].reshape(-1))
                    self.std.copy_(mean_std[1].reshape(-1))
                else:
                    self.register_buffer("mean", mean_std[0].reshape(-1).clone())
                    self.register_buffer("std", mean_std[1].reshape(-1).clone())
        self.codebook_freq = torch.zeros(quantizer.codebook_size, dtype=torch.long)
        self.model_type = model_type
        if self.model_type == "transformer":
            if num_latent_tokens is not None:
                scale = self.encoder.width**-0.5
                self.latent_tokens = nn.Parameter(
                    scale * torch.randn(num_latent_tokens, self.encoder.width)
                )
        self._load_ckpt()

    def _load_ckpt(self):
        raise NotImplementedError("Please implement this method in the subclass")

    @torch.no_grad()
    def index_to_latent(self, indices: torch.Tensor) -> torch.Tensor:
        with (
            torch.autocast(device_type="cuda", dtype=self.fwd_dtype)
            if self.fwd_dtype not in [torch.float32]
            else nullcontext()
        ):
            index_shape = indices.shape
            index = einops.rearrange(indices, "... -> 1 (...)")
            if isinstance(self.quantizer, FSQ):
                latent = self.quantizer.indices_to_codes(index)
            elif isinstance(self.quantizer, VectorQuantize):
                latent = self.quantizer.get_output_from_indices(index)
            else:
                raise ValueError(f"Unsupported quantizer type: {type(self.quantizer)}")
            latent = latent.reshape(*index_shape, -1)
            if latent.ndim == 4:
                latent = einops.rearrange(latent, "b h w c -> b c h w")
            elif latent.ndim == 3:
                latent = einops.rearrange(latent, "h w c -> 1 c h w")
            elif latent.ndim == 5:
                latent = einops.rearrange(latent, "b t h w c -> b c t h w")
            else:
                raise ValueError(f"Unsupported latent shape: {latent.shape}")
            return latent


class VideoVQ(TowDVQ):

    def __init__(
        self,
        model_type: str = "transformer",
        train2d: bool = False,
        train3d: bool = True,
        forward_video: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(model_type=model_type, *args, **kwargs)
        # model type can be 'transformer' or '3dcnn'
        self.train2d = train2d
        self.train3d = train3d
        self.train_2d3d = train2d and train3d
        assert not self.train_2d3d, "暂时不支持2d和3d同时训练"
        self.forward_video = forward_video

    def _get_visual(self, inputs: dict) -> torch.Tensor | dict:
        if self.train_2d3d:
            result_dict = {}
            visual_3d = [x for x in inputs["video"] if x is not None]
            if len(visual_3d) == 0:
                visual_3d = None
            else:
                visual_3d = torch.stack(visual_3d, dim=0)  # [B, T, C, H, W]
            visual_2d = [x for x in inputs["image"] if x is not None]
            if len(visual_2d) == 0:
                visual_2d = None
            else:
                visual_2d = torch.stack(visual_2d, dim=0)  # [B, C, H, W]
                visual_2d = einops.rearrange(visual_2d, "b c h w -> b 1 c h w")
            result_dict["video"] = visual_3d
            result_dict["image"] = visual_2d
            return result_dict
        if self.train3d:
            if isinstance(inputs["video"], torch.Tensor):
                visual = inputs["video"]
                if visual.ndim == 4:
                    visual = einops.rearrange(visual, "... -> 1 ...")
                assert (
                    visual.ndim == 5
                ), f"video shape should be [B, T, C, H, W], but got {visual.shape}"
                return visual
            visual = [x for x in inputs["video"] if x is not None]
            visual = torch.stack(visual, dim=0)  # [B, T, C, H, W]
            return visual
        visual = super()._get_visual(inputs)
        visual = einops.rearrange(visual, "b c h w -> b 1 c h w")
        return visual

    def _load_ckpt(self):
        if self.ckpt_path is not None:
            assert os.path.exists(
                self.ckpt_path
            ), f"Please provide the correct path of the weight of the vq, the path you provide is: {self.ckpt_path}"
            state_dict = load_file(self.ckpt_path)
            self.load_state_dict(state_dict)
        if self.freeze:
            freeze_model(self)

    @torch.no_grad()
    def encode_to_index(
        self, video: Tensor | None = None, features: Tensor | None = None
    ) -> Tensor:
        # 提取特征
        if video is not None:
            B, T = video.shape[:2]
        else:
            assert (
                features is not None
            ), "video and features should not be None at the same time"
            B, T = features.shape[:2]
        features = self._extra_features({"video": video, "features": features})
        # norm features
        features = self.norm_features(features)
        features = features.to(self.fwd_dtype)
        tensor_var = features
        # 编码
        with (
            maybe_autocast(tensor_var, self.fwd_dtype)
            if self.fwd_dtype not in [torch.float32]
            else nullcontext()
        ):
            # 编码
            x = self.vq_encode(features, forward_T=T)
            # 量化
            x = einops.rearrange(x, "b c ... -> b (...) c")
            if self.commit_loss_weight > 0:
                quantized, indices, commit_loss = self.quantizer(x.float())
            else:
                quantized, indices = self.quantizer(x.float())
        return quantized, indices  # B(H W)C, B(H W) or B(T H W)C, B(T H W)

    def _extra_features(self, inputs: dict):
        if inputs.get("features", None) is not None:
            features = inputs["features"]
            if isinstance(features, list):
                features = torch.stack(features, dim=0)
            return features

        video = inputs["video"]
        B, T = video.shape[:2]
        if self.forward_video:
            features = self.feature_extractor(video)
        else:
            video = einops.rearrange(video, "b t c h w -> (b t) c h w")
            features = self.feature_extractor(video)
            features = einops.rearrange(features, "(b t) c h w -> b t c h w", b=B)
        return features

    def norm_features(self, features):
        if self.mean_std_path is not None:
            features = einops.rearrange(features, "b t c h w -> b t h w c")
            features = (features - self.mean) / (self.std + 1e-8)
            features = einops.rearrange(features, "b t h w c -> b t c h w")
        return features

    def denorm_features(self, features):
        if self.mean_std_path is not None:
            features = einops.rearrange(features, "b t c h w -> b t h w c")
            features = features * (self.std + 1e-8) + self.mean
            features = einops.rearrange(features, "b t h w c -> b t c h w")
        return features

    def vq_encode(self, video: Tensor, forward_T: int):
        if self.model_type == "transformer":
            return self.encoder(video, forward_T=forward_T)  # [B, C, H, W]
        if self.model_type == "3dcnn":
            assert (
                video.ndim == 5
            ), f"video shape should be [B, T, C, H, W], but got {video.shape}"
            if video.shape[1] == 1:
                video = video[:, 0]
            else:
                video = einops.rearrange(video, "b t c h w -> b c t h w")
            feature = self.encoder(video)  # [B, C, T, H, W] or [B, C, H, W]
            return feature
        raise ValueError(f"Unknown model type: {self.model_type}")

    @torch.no_grad()
    def index_to_feature(
        self, index: torch.Tensor, denormalize=True, forward_T: int | None = None
    ) -> torch.Tensor:
        latent = self.index_to_latent(index)
        # latent = einops.rearrange(latent, '... -> 1 ...')
        latent = latent.to(self.fwd_dtype)
        with (
            maybe_autocast(latent, self.fwd_dtype)
            if self.fwd_dtype not in [torch.float32]
            else nullcontext()
        ):
            features = self.vq_decode(latent, forward_T=forward_T)
        if denormalize:
            features = self.denorm_features(features)
        return features

    def vq_decode(self, quantized: Tensor, forward_T: int | None = None):
        if self.model_type == "transformer":
            return self.decoder(quantized, forward_T=forward_T)
        if self.model_type == "3dcnn":
            feature = self.decoder(quantized)  # [B, C, T, H, W] or [B, C, H, W]
            if feature.ndim == 4:
                feature = einops.rearrange(feature, "b c h w -> b 1 c h w")
            else:
                feature = einops.rearrange(feature, "b c t h w -> b t c h w")
            return feature
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _inner_forward(self, inputs: dict, **kwargs):
        # 提取特征
        # videos = inputs['video']
        # B, T = videos.shape[:2]
        features = self._extra_features(inputs)
        B, T = features.shape[:2]
        features = self.norm_features(features)
        features = features.to(self.fwd_dtype)
        tensor_var = features
        with (
            maybe_autocast(tensor_var, self.fwd_dtype)
            if self.fwd_dtype not in [torch.float32]
            else nullcontext()
        ):
            # 编码
            x = self.vq_encode(features, forward_T=T)

            # 量化
            x_shape = x.shape
            x = einops.rearrange(x, "b c ... -> b (...) c")
            if self.commit_loss_weight > 0:
                quantized, indices, commit_loss = self.quantizer(x.float())
            else:
                quantized, indices = self.quantizer(x.float())
                commit_loss = torch.zeros(1, device=quantized.device)

            quantized = einops.rearrange(quantized, "b ... c -> b c ...")
            quantized = quantized.reshape(x_shape)

            # 解码
            re_features = self.vq_decode(quantized, forward_T=T)
            if re_features.ndim == 4:
                re_features = einops.rearrange(re_features, "b c h w -> b 1 c h w")
        return (
            features.to(self.fwd_dtype),
            re_features.to(self.fwd_dtype),
            commit_loss,
            indices,
        )

    def forward(self, inputs: dict, return_features: bool = False, **kwargs):
        videos = None
        if videos is None:
            if isinstance(inputs, dict):
                if "video" in inputs:
                    video = inputs["video"]
                    if isinstance(video, list):
                        all_video_not_none = [x for x in video if x is not None]
                        if len(all_video_not_none) > 0:
                            videos = self._get_visual(inputs)
                        else:
                            videos = None
                    elif isinstance(video, torch.Tensor):
                        videos = video
                    else:
                        raise ValueError(
                            f"video should be a list, but got {type(video)}"
                        )
                else:
                    videos = None
            else:
                videos = inputs  # [B,T,C,H,W], uint8
                inputs = {}
        inputs["video"] = videos
        self._quantizer_force_float()
        features, re_features, commit_loss, indices = self._inner_forward(inputs)
        # loss
        re_loss = self.re_loss_fn(
            einops.rearrange(features, "b t c h w -> (b t h w) c"),
            einops.rearrange(re_features, "b t c h w -> (b t h w) c"),
        ).mean()
        IFrame_loss = self.re_loss_fn(
            einops.rearrange(features[:, 0].detach(), "b c h w -> b (h w) c"),
            einops.rearrange(re_features[:, 0].detach(), "b c h w -> b (h w) c"),
        ).mean()
        if features.shape[1] > 1:
            PFrame_loss = self.re_loss_fn(
                einops.rearrange(features[:, 1:].detach(), "b t c h w -> b (t h w) c"),
                einops.rearrange(
                    re_features[:, 1:].detach(), "b t c h w -> b (t h w) c"
                ),
            ).mean()
        else:
            PFrame_loss = torch.zeros(1, device=features.device)
        if return_features:
            return {
                "re_features": re_features,
                "features": features,
                "commit_loss": commit_loss.mean(),
                "re_loss": re_loss,
            }
        commit_loss = commit_loss.mean()
        loss = self.commit_loss_weight * commit_loss + self.recon_loss_weight * re_loss

        losses = dict(total_loss=loss)

        return losses
