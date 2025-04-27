from dataclasses import dataclass, fields
from pathlib import Path

import fiddle as fdl
import torch
from fiddle import Config
from safetensors.torch import load_file

from landiff.llm.models.lm_model import Semantic1DLM
from landiff.utils import freeze_model, set_seed_for_single_process


@dataclass
class ARSampleCfg:
    top_k: int | None = None
    top_p: float | None = None
    temperature: float = 1.0
    teacher_forcing: bool = False
    use_gt_first_frame: bool = False
    cfg: float = 0.0
    motion_score: float | None = None
    num_frames: int = 13  # 49 RGB frames, 13 semantic frames

    def __str__(self):
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value != field.default:
                parts.append(f"{field.name}_{value}")
        if not parts:
            return "default"
        return ",".join(parts)

    # to dict,默认值不会被保存
    def to_dict(self):
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) != field.default
        }

    def to_dict_str(self):
        obj_dict = self.to_dict()
        dict_str = str(obj_dict)
        dict_str = dict_str.replace(" ", "")
        return dict_str


@dataclass
class CodeTask:
    save_file_name: str
    prompt: str
    seed: int
    result: None | torch.Tensor = None
    sample_cfg: ARSampleCfg = ARSampleCfg()


class ArModelInferWrapper(torch.nn.Module):

    def __init__(self, ckpt_path: str, model_cfg: Config):
        super().__init__()
        self.config = model_cfg

        self.model: Semantic1DLM = fdl.build(self.config)
        assert isinstance(self.model, Semantic1DLM), "model is not a Semantic1DLM"
        assert Path(ckpt_path).exists(), f"ckpt_path: {ckpt_path} does not exist"
        assert (
            Path(ckpt_path).suffix == ".safetensors"
        ), f"ckpt_path: {ckpt_path} is not a safetensors file"
        ckpt_state = load_file(ckpt_path)
        self.model.load_state_dict(ckpt_state, strict=True)
        freeze_model(self.model)

    @torch.no_grad()
    def forward(self, code_task: CodeTask) -> CodeTask:
        sample_cfg = code_task.sample_cfg
        input = {}
        input["caption"] = [code_task.prompt]
        if sample_cfg.motion_score is not None:
            # print('use motion_score:', sample_cfg.motion_score)
            motion_score = torch.tensor(
                [sample_cfg.motion_score], dtype=torch.float32, device="cuda"
            )
            input["motion_score"] = motion_score
        else:
            input["motion_score"] = None

        input["frames"] = torch.tensor(
            [sample_cfg.num_frames], dtype=torch.float32, device="cuda"
        )
        set_seed_for_single_process(code_task.seed)
        semantic_token = self.model.sample(
            input,
            teacher_forcing=sample_cfg.teacher_forcing,
            seed=code_task.seed,
            top_k=sample_cfg.top_k,
            guidance_scale=sample_cfg.cfg,
            top_p=sample_cfg.top_p,
            temperature=sample_cfg.temperature,
            use_gt_first_frame=sample_cfg.use_gt_first_frame,
            num_frames=sample_cfg.num_frames,
        )
        semantic_token = semantic_token.cpu().reshape(-1)
        code_task.result = semantic_token
        return code_task
