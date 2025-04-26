from pathlib import Path

import numpy as np
import torch

from landiff.diffusion.dif_infer import CogModelInferWrapper, VideoTask
from landiff.llm.llm_cfg import build_llm
from landiff.llm.llm_infer import ArModelInferWrapper, ARSampleCfg, CodeTask
from landiff.utils import save_video_tensor


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Landiff Video Inference")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the video generation.",
    )
    parser.add_argument(
        "--llm_ckpt",
        type=str,
        help="Path to the LLM checkpoint.",
        default="ckpts/LanDiff/llm/model.safetensors",
    )
    parser.add_argument(
        "--diffusion_ckpt",
        type=str,
        help="Path to the diffusion checkpoint.",
        default="ckpts/LanDiff/diffusion",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        help="Path to save the generated video.",
        default="results/video",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="CFG scale for the video generation.",
    )
    parser.add_argument(
        "--motion_score",
        type=float,
        default=0.1,
        help="Motion score for the video generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video generation.",
    )

    return parser.parse_args()


def llm_infer(args):
    llm_model_cfg = build_llm()
    llm_mode = ArModelInferWrapper(args.llm_ckpt, llm_model_cfg)
    llm_mode = llm_mode.cuda()
    code_task = CodeTask(
        save_file_name=f"{args.save_file_name}.npy",
        prompt=args.prompt,
        seed=args.seed,
        sample_cfg=ARSampleCfg(
            temperature=1.0,
            cfg=args.cfg,
            motion_score=args.motion_score,
        ),
    )
    code_task: CodeTask = llm_mode(code_task)
    semantic_token = code_task.result.reshape(-1)
    semantic_save_path = Path(code_task.save_file_name)
    semantic_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        semantic_save_path,
        semantic_token.cpu().numpy(),
    )
    llm_mode = llm_mode.cpu()
    torch.cuda.empty_cache()
    semantic_token = semantic_token.cuda()
    return semantic_token


def infer_diffusion(args, semantic_token):
    diffusion_model = CogModelInferWrapper(ckpt_path=args.diffusion_ckpt)
    diffusion_model = diffusion_model.cuda()
    video_task = VideoTask(
        save_file_name=f"{args.save_file_name}.mp4",
        prompt=args.prompt,
        seed=args.seed,
        fps=8,
        semantic_token=semantic_token,
    )
    video_task: VideoTask = diffusion_model(video_task)
    video = video_task.result
    save_video_tensor(video, video_task.save_file_name, fps=video_task.fps)
    print(f"save video to {video_task.save_file_name}")


def main():
    import os

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    semantic_token = llm_infer(args)

    infer_diffusion(args, semantic_token)


if __name__ == "__main__":
    main()
