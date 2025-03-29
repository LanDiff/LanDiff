from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig
from sat import mpu
from sat.helpers import print_rank0

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...util import append_dims, instantiate_from_config


class ExtraLossRegistry:
    _registry = {}  # name -> weakref of dataset

    def __init__(self):
        raise TypeError("StatefulDatasetRegistry should not be instantiated.")

    @staticmethod
    def register(name: str, loss: torch.Tensor):
        assert isinstance(name, str), name
        assert isinstance(loss, torch.Tensor), loss
        ExtraLossRegistry._registry[name] = loss

    @staticmethod
    def get_loss(name: str):
        return ExtraLossRegistry._registry.get(name, None)

    @staticmethod
    def clear():
        ExtraLossRegistry._registry.clear()


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise
                + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim)
                * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(
        self,
        block_scale=None,
        block_size=None,
        min_snr_value=None,
        fixed_frames=0,
        extra_loss_weight: dict = {},
        # prefix_length=0,
        **kwargs,
    ):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        self.extra_loss_weight = extra_loss_weight
        # self.prefix_length = prefix_length
        print_rank0(f"loss extra_loss_weight: {extra_loss_weight}")
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(
            noise, src=src, group=mpu.get_model_parallel_group()
        )
        torch.distributed.broadcast(
            alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group()
        )

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise
                + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim)
                * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(
            alphas_cumprod_sqrt, input.ndim
        ) + noise * append_dims((1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim)

        if self.fixed_frames > 0:
            # replace the prefix with the original input
            noised_input[:, : self.fixed_frames] = input[:, : self.fixed_frames]
        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]
        additional_model_inputs["vae_featrues"] = input
        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        if self.fixed_frames > 0:
            # remove the prefix for loss calculation,only calculate the loss for the generated part
            input = input[:, self.fixed_frames :]
            model_output = model_output[:, self.fixed_frames :]
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            loss = torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            loss = torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
        loss_dict = {
            "diffusion_loss": loss.mean().detach().clone(),
        }
        totol_loss = loss.mean()
        commit_loss = ExtraLossRegistry.get_loss("commit_loss")
        if commit_loss is not None:
            loss_dict["commit_loss"] = commit_loss.mean().detach().clone()
            totol_loss = (
                totol_loss + commit_loss * self.extra_loss_weight["commit_loss"]
            )

        re_loss = ExtraLossRegistry.get_loss("re_loss")
        if re_loss is not None:
            loss_dict["re_loss"] = re_loss.mean().detach().clone()
            totol_loss = totol_loss + re_loss * self.extra_loss_weight["re_loss"]
        loss_dict["total_loss"] = totol_loss
        return loss_dict
