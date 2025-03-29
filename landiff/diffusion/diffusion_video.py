import gc
import math
import random
from typing import Any, Dict, List, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from sat import mpu
from sat.helpers import print_rank0
from torch import nn

from landiff.diffusion.sgm.modules import UNCONDITIONAL_CONFIG
from landiff.diffusion.sgm.modules.autoencoding.temporal_ae import VideoDecoder
from landiff.diffusion.sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from landiff.diffusion.sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)

from .dit_video_concat import ControlDiffWarp


class SATVideoDiffusionEngine(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get(
            "disable_first_stage_autocast", False
        )
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get(
            "not_trainable_prefixes", ["first_stage_model", "conditioner"]
        )
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get(
            "en_and_decode_n_samples_a_time", None
        )
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get(
            "noised_image_all_concat", False
        )
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )

        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def disable_untrainable_params(self):
        total_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            flag = False
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all":
                    flag = True
                    break

            lora_prefix = ["matrix_A", "matrix_B"]
            for prefix in lora_prefix:
                if prefix in n:
                    flag = False
                    break

            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()

        print_rank0(
            "***** Total trainable parameters: " + str(total_trainable) + " *****"
        )

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def forward(self, x, batch):
        loss: dict = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        if "total_loss" in loss:
            total_loss = loss["total_loss"]
        else:
            total_loss = sum(loss.values())
        # loss_dict = {"loss": loss_mean}
        return total_loss, loss

    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)  # shape [b,t,c,h,w]
        if self.lr_scale is not None:
            lr_x = F.interpolate(
                x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False
            )
            lr_x = F.interpolate(
                lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False
            )
            lr_z = self.encode_first_stage(lr_x, batch)
            batch["lr_input"] = lr_z

        x = einops.rearrange(
            x, "b t c h w -> b c t h w"
        ).contiguous()  # x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.noised_image_input:
            image = x[:, :, 0:1]
            image = self.add_noise_to_first_frame(image)
            image = self.encode_first_stage(image, batch)
        # gc.collect()
        # torch.cuda.empty_cache()
        x = self.encode_first_stage(x, batch)
        x = einops.rearrange(
            x, "b c t h w -> b t c h w"
        ).contiguous()  # x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.noised_image_input:
            image = einops.rearrange(
                image, "b c t h w -> b t c h w"
            ).contiguous()  # image = image.permute(0, 2, 1, 3, 4).contiguous()
            if self.noised_image_all_concat:
                image = image.repeat(1, x.shape[1], 1, 1, 1)
            else:
                image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
            if random.random() < self.noised_image_dropout:
                image = torch.zeros_like(image)
            batch["concat_images"] = image

        # gc.collect()
        # torch.cuda.empty_cache()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        # x shape [b,c,t,h,w]
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.contiguous()  # [b,c,t,h,w]
            return x * self.scale_factor  # already encoded

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        # z shape [b,c,t,h,w]
        return z

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        prefix=None,
        concat_images=None,
        generator: None | torch.Generator | list[torch.Generator] = None,
        **kwargs,
    ):
        if generator is None or isinstance(generator, torch.Generator):
            randn = torch.randn(
                batch_size,
                *shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            randn = torch.stack(
                [
                    torch.randn(
                        *shape, generator=g, device=self.device, dtype=torch.float32
                    )
                    for g in generator
                ]
            )
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        if prefix is not None:  # add prefix video frames to the noise as condition
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(
                randn, src=src, group=mpu.get_model_parallel_group()
            )

        scale = None
        scale_emb = None

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model,
            input,
            sigma,
            c,
            concat_images=concat_images,
            **addtional_model_inputs,
        )

        samples = self.sampler(
            denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb, **kwargs
        )
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        only_log_video_latents=False,
        **kwargs,
    ) -> Dict:
        gc.collect()
        torch.cuda.empty_cache()
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=(
                ucg_keys if len(self.conditioner.embedders) > 0 else []
            ),
        )

        sampling_kwargs = {}
        sampling_kwargs["mp4"] = batch["mp4"]
        if "cog_vae_embedding" in batch:
            sampling_kwargs["vae_featrues"] = batch["cog_vae_embedding"]

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        # if not self.latent_input:
        #     log["inputs"] = x.to(torch.float32)
        if "mp4" in batch:
            log["inputs"] = batch["mp4"][:N].to(self.device).to(torch.float32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = (
                log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
            )
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if self.noised_image_input:
            image = x[:, :, 0:1]
            image = self.add_noise_to_first_frame(image)
            image = self.encode_first_stage(image, batch)
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
            c["concat"] = image
            uc["concat"] = image
            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )  # b t c h w
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        else:
            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )  # b t c h w
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        gc.collect()
        torch.cuda.empty_cache()
        return log


class SATControlVideoDiffusionEngine(SATVideoDiffusionEngine):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        model_config = args.model_config
        network_config = model_config.get("network_config", None)
        control_network_config = model_config.get("control_network_config", None)
        assert control_network_config is not None, "control_network_config is required"
        control_network_config["params"]["dtype"] = self.dtype_str
        network_wrapper = model_config.get("network_wrapper", None)
        network_config["params"]["dtype"] = self.dtype_str
        model = instantiate_from_config(network_config)

        compile_model = model_config.get("compile_model", False)
        model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=self.dtype
        )
        control_model = instantiate_from_config(control_network_config)
        control_model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            control_model, compile_model=compile_model, dtype=self.dtype
        )
        pretrain_diffusion_model_ckpt_path = model_config.get(
            "pretrain_diffusion_model_ckpt_path", None
        )
        assert (
            pretrain_diffusion_model_ckpt_path is not None
        ), "pretrain_diffusion_model_ckpt_path is required"
        freeze_dit = model_config.get("freeze_dit", True)
        self.model = ControlDiffWarp(
            main_model=model,
            control_model=control_model,
            pretrain_diffusion_model_ckpt_path=pretrain_diffusion_model_ckpt_path,
            freeze_dit=freeze_dit,
        )
