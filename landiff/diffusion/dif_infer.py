import argparse
import math
import shlex
from dataclasses import dataclass

import einops
import numpy as np
import torch
from omegaconf import ListConfig
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint

from landiff.diffusion.arguments import get_args
from landiff.diffusion.diffusion_video import (
    SATControlVideoDiffusionEngine,
    SATVideoDiffusionEngine,
)
from landiff.diffusion.sgm.util import InferValueRegistry
from landiff.utils import freeze_model, set_seed_for_single_process, stable_hash


def _pre_process_cog_video(video: torch.Tensor) -> torch.Tensor:
    """
    Pre-processes the input video tensor by scaling its values.

    Args:
        video (torch.Tensor): The input video tensor to be pre-processed.

    Returns:
        torch.Tensor: The pre-processed video tensor with values scaled to the range [-1, 1].
    """
    video = video * 2.0 - 1.0
    video = torch.clamp(video, -1.0, 1.0)
    return video


def _post_process_cog_video(video: torch.Tensor) -> torch.Tensor:
    """
    Post-processes the input video tensor by scaling its values.

    Args:
        video (torch.Tensor): The input video tensor to be post-processed.

    Returns:
        torch.Tensor: The post-processed video tensor with values scaled to the range [0, 1].
    """
    video = (video + 1.0) / 2.0
    video = torch.clamp(video, 0.0, 1.0)
    return video


def get_batch(keys, value_dict, N: list | ListConfig, T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


@dataclass
class CogOutput:
    video: torch.Tensor  # rgb video tensor, shape [B, C, T, H, W], range [0,1]
    latent: torch.Tensor  # latent tensor, shape [B, T, C, H, W]


@dataclass
class VideoTask:
    save_file_name: str
    prompt: str
    seed: int
    fps: int = 8
    mp4: None | torch.Tensor = None
    semantic_token: None | torch.Tensor = None
    result: None | torch.Tensor = None


class CogWrapper(torch.nn.Module):

    def __init__(
        self,
        args_str: str,
        fwd_dtype: torch.dtype = torch.bfloat16,
        image_size: list[int] = [480, 720],
        seed: int | None = None,
        engine_type: str = "control",  # control or normal
        **kwargs,
    ):
        super().__init__()
        self.args_str = args_str
        # args str to list
        args = shlex.split(args_str)
        py_parser = argparse.ArgumentParser(add_help=False)
        known, args_list = py_parser.parse_known_args()
        known, args_list = py_parser.parse_known_args(args)
        args = get_args(args_list, diff_seed_for_each_gpu=False, init_distributed=False)
        args = argparse.Namespace(**vars(args), **vars(known))
        del args.deepspeed_config
        args.model_config.first_stage_config.params.cp_size = 1
        args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        args.model_config.network_config.params.transformer_args.checkpoint_activations = (
            False
        )
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = (
            False
        )

        # config
        self.args = args
        self.fwd_dtype = fwd_dtype
        self.image_size = image_size
        self.seed = self.args.seed
        if seed is not None:
            self.seed = seed
        self.engine_type = engine_type
        if self.engine_type == "normal":
            model: SATVideoDiffusionEngine = get_model(args, SATVideoDiffusionEngine)
        elif self.engine_type == "control":
            model: SATControlVideoDiffusionEngine = get_model(
                args, SATControlVideoDiffusionEngine
            )
        else:
            raise ValueError(f"Unknown engine type: {self.engine_type}")
        load_checkpoint(model, args)
        self.model = model
        self.model = self.model.to(self.fwd_dtype)
        freeze_model(self.model)

    @torch.no_grad()
    def forward(
        self,
        inputs: dict,
        seed: int | None = None,
        semantic_token: torch.Tensor | None = None,
        semantic_feature_before_upsample: torch.Tensor | None = None,
        vae_feature_prefix: torch.Tensor | None = None,
    ) -> CogOutput:
        InferValueRegistry.clear()  # 清理缓存的semantic features
        if semantic_token is not None:
            semantic_token = semantic_token.cuda()
            semantic_token = einops.rearrange(semantic_token, "... -> 1 1 (...)")
            InferValueRegistry.register("semantic_token", semantic_token)
        if semantic_feature_before_upsample is not None:
            semantic_feature_before_upsample = semantic_feature_before_upsample.cuda()
            InferValueRegistry.register(
                "semantic_feature_before_upsample", semantic_feature_before_upsample
            )
        image_size = self.image_size
        sample_func = self.model.sample
        T, H, W, C, F = (
            self.args.sampling_num_frames,
            image_size[0],
            image_size[1],
            self.args.latent_channels,
            8,
        )
        num_samples = [1]
        force_uc_zero_embeddings = ["txt"]
        # get data
        text = inputs["caption"]
        mp4 = inputs["video"]  # B C T H W range [0, 1]
        if mp4 is not None:
            mp4 = einops.rearrange(
                mp4, "b c t h w -> b t c h w"
            )  # cog need [T, C, H, W]
            mp4 = _pre_process_cog_video(mp4)
        text_hash = stable_hash(str(text))
        if seed is not None:
            text_seed = seed
        else:
            text_seed = (text_hash + self.seed) % 2**32
        set_seed_for_single_process(text_seed)
        value_dict = {
            "prompt": text,
            "negative_prompt": "",
            "num_frames": torch.tensor(T).unsqueeze(0),
        }

        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.model.conditioner),
            value_dict,
            num_samples,
        )
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                print(key, batch[key].shape)
            elif isinstance(batch[key], list):
                print(key, [len(l) for l in batch[key]])
            else:
                print(key, batch[key])
        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=force_uc_zero_embeddings,
        )

        for k in c:
            if not k == "crossattn":
                c[k], uc[k] = map(
                    lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                )

        samples_z = sample_func(
            c,
            uc=uc,
            batch_size=1,
            shape=(T, C, H // F, W // F),
            # generator=generator,
            prefix=vae_feature_prefix,
            mp4=mp4,
        )
        recon = self.decode_latent(
            einops.rearrange(samples_z, "b t c h w -> b c t h w")
        )
        samples = _post_process_cog_video(recon)
        output = CogOutput(
            video=samples,
            latent=samples_z,
        )
        return output

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor):
        self.model.to("cpu")
        first_stage_model = self.model.first_stage_model
        torch.cuda.empty_cache()
        first_stage_model = first_stage_model.cuda()
        latent = 1.0 / self.model.scale_factor * latent  # shape [b, c, t, h, w]
        T = latent.shape[2]
        # Decode latent serial to save GPU memory
        recons = []
        loop_num = (T - 1) // 2
        for i in range(loop_num):
            if i == 0:
                start_frame, end_frame = 0, 3
            else:
                start_frame, end_frame = i * 2 + 1, i * 2 + 3
            if i == loop_num - 1:
                clear_fake_cp_cache = True
            else:
                clear_fake_cp_cache = False
            recon = first_stage_model.decode(
                latent[:, :, start_frame:end_frame].contiguous(),
                clear_fake_cp_cache=clear_fake_cp_cache,
            )
            recons.append(recon.cpu())
        recon = torch.cat(recons, dim=2).to(torch.float32)
        return recon


class CogModelInferWrapper(torch.nn.Module):

    def __init__(
        self,
        ckpt_path: str,
        infer_cfg_path: str = "landiff/diffusion/configs/infer_cfgs/2b.yaml",
        model_cfg_path: str = "landiff/diffusion/configs/cogvideox_2b_control_theia_interpolate_video_vq.yaml",
    ):
        super().__init__()
        self.infer_cfg_path = infer_cfg_path
        self.model_cfg_path = model_cfg_path
        self.ckpt_path = ckpt_path
        args_str = f"--base {model_cfg_path} {infer_cfg_path} --load {ckpt_path}"
        self.init_infer_model = CogWrapper(args_str=args_str)

    @torch.no_grad()
    def forward(self, x: VideoTask):
        inputs = dict(
            caption=x.prompt,
            video=x.mp4.cuda() if x.mp4 is not None else None,
        )
        output: CogOutput = self.init_infer_model.forward(
            inputs, seed=x.seed, semantic_token=x.semantic_token
        )  # shape B, C, T, H, W, range [0, 1]
        video = output.video
        assert video.shape[0] == 1, f"video.shape[0] != 1, {video.shape[0]}"
        video = video.cpu()
        x.result = video[0]
        return x
