import os
import sys
import tempfile

import torch

import gradio as gr

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landiff.diffusion.dif_infer import CogModelInferWrapper, VideoTask
from landiff.llm.llm_cfg import build_llm
from landiff.llm.llm_infer import ArModelInferWrapper, ARSampleCfg, CodeTask
from landiff.utils import save_video_tensor


# Initialize models
def initialize_models(llm_ckpt_path, diffusion_ckpt_path):
    print("📝 Initializing LLM model...")
    llm_model_cfg = build_llm()
    llm_model = ArModelInferWrapper(llm_ckpt_path, llm_model_cfg)

    print("🎬 Initializing diffusion model...")
    diffusion_model = CogModelInferWrapper(ckpt_path=diffusion_ckpt_path)

    return llm_model, diffusion_model


# Generate video
def generate_video(
    prompt, cfg_scale, motion_score, llm_model, diffusion_model, seed=42
):
    print(f"🚀 Processing prompt: {prompt}")
    print(f"🎲 Using random seed: {seed}")

    # Move LLM model to GPU
    llm_model = llm_model.cuda()

    # Create temporary save path, but don't auto-delete
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    save_path = os.path.join(temp_dir, "output")

    # Use LLM to generate semantic tokens
    code_task = CodeTask(
        save_file_name=f"{save_path}.npy",
        prompt=prompt,
        seed=seed,
        sample_cfg=ARSampleCfg(
            temperature=1.0,
            cfg=cfg_scale,
            motion_score=motion_score,
        ),
    )

    print("🧠 LLM generating semantic tokens...")
    code_task = llm_model(code_task)
    semantic_token = code_task.result.reshape(-1)

    # Move LLM back to CPU to free GPU memory
    llm_model = llm_model.cpu()
    torch.cuda.empty_cache()

    # Move diffusion model to GPU
    diffusion_model = diffusion_model.cuda()
    semantic_token = semantic_token.cuda()

    # Generate video using semantic tokens
    video_path = f"{save_path}.mp4"
    video_task = VideoTask(
        save_file_name=video_path,
        prompt=prompt,
        seed=seed,
        fps=8,
        semantic_token=semantic_token,
    )

    print("🎥 Diffusion model generating video...")
    video_task = diffusion_model(video_task)
    video = video_task.result

    # Save video
    save_video_tensor(video, video_path, fps=video_task.fps)
    print(f"✅ Video generation completed, saved to: {video_path}")

    # Move diffusion model back to CPU, free GPU memory
    diffusion_model = diffusion_model.cpu()
    torch.cuda.empty_cache()

    return video_path


# Gradio interface
def create_ui(llm_model, diffusion_model):
    with gr.Blocks(
        title="LanDiff: Text-to-Video Generation", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # LanDiff: Integrating Language Models and Diffusion Models for Video Generation
            
            Provide a detailed text description, and LanDiff will generate a corresponding video for you.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Please enter a detailed scene description, for example: A snail with a brown and tan shell is seen crawling on a bed of green moss. The snail's body is grayish-brown, and it has two prominent tentacles extended forward. The environment suggests a natural, outdoor setting with a focus on the snail's movement across the mossy surface.",
                    lines=5,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            info="Controls the influence of the text prompt on the generated content",
                        )

                    with gr.Column(scale=1):
                        motion_score = gr.Slider(
                            label="Motion Score",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            info="Controls the degree of motion in the video",
                        )

                with gr.Row():
                    seed = gr.Number(
                        label="Random Seed",
                        value=42,
                        precision=0,
                        info="Set a random seed to get reproducible results",
                    )
                    random_seed_btn = gr.Button("Random Seed", size="sm")

                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Generated Video")
                status = gr.Markdown("Waiting for generation...")

        def init_video():
            return "Waiting for generation...", None

        # Process functions
        def process(prompt, cfg, motion, seed_value):
            try:
                # yield "🔄 Generating video, please wait...", None

                video_path = generate_video(
                    prompt, cfg, motion, llm_model, diffusion_model, int(seed_value)
                )
                print("✅ Video generation successful!")
                return "✅ Video generation successful!", video_path
            except Exception as e:
                print(f"❌ Generation failed: {str(e)}")
                return f"❌ Generation failed: {str(e)}", None

        def get_random_seed():
            import random

            return int(random.randint(0, 2**31 - 1))

        random_seed_btn.click(fn=get_random_seed, inputs=[], outputs=[seed])

        generate_btn.click(
            fn=init_video, outputs=[status, video_output], inputs=[]
        ).then(
            fn=process,
            inputs=[prompt, cfg_scale, motion_score, seed],
            outputs=[status, video_output],
        )

        gr.Markdown(
            """
            ---
            ### About LanDiff
            LanDiff is a video generation system that integrates language models and diffusion models to generate high-quality video content from detailed text descriptions.
            
            [Project Page](https://landiff.github.io/) | [GitHub](https://github.com/LanDiff/LanDiff) | [Paper](https://arxiv.org/abs/2503.04606)
            """
        )

    return demo


# Main function
def main():
    # Model path settings
    llm_ckpt_path = "ckpts/LanDiff/llm/model.safetensors"
    diffusion_ckpt_path = "ckpts/LanDiff/diffusion"

    print("🔧 Setting up environment...")
    # Initialize models (first on CPU)
    llm_model, diffusion_model = initialize_models(llm_ckpt_path, diffusion_ckpt_path)

    # Create and launch Gradio interface
    demo = create_ui(llm_model, diffusion_model)
    demo.queue(max_size=10).launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
