import os
import sys
import tempfile

import torch

import gradio as gr

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landiff.diffusion.dif_infer import CogModelInferWrapper, VideoTask
from landiff.llm.llm_cfg import build_llm
from landiff.llm.llm_infer import ArModelInferWrapper, ARSampleCfg, CodeTask
from landiff.utils import save_video_tensor


# 初始化模型
def initialize_models(llm_ckpt_path, diffusion_ckpt_path):
    print("📝 正在初始化LLM模型...")
    llm_model_cfg = build_llm()
    llm_model = ArModelInferWrapper(llm_ckpt_path, llm_model_cfg)

    print("🎬 正在初始化扩散模型...")
    diffusion_model = CogModelInferWrapper(ckpt_path=diffusion_ckpt_path)

    return llm_model, diffusion_model


# 生成视频
def generate_video(
    prompt, cfg_scale, motion_score, llm_model, diffusion_model, seed=42
):
    print(f"🚀 开始处理提示: {prompt}")
    print(f"🎲 使用随机种子: {seed}")

    # 将LLM模型移到GPU
    llm_model = llm_model.cuda()

    # 创建临时保存路径，但不自动删除
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")

    save_path = os.path.join(temp_dir, "output")

    # 使用LLM生成语义令牌
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

    print("🧠 LLM生成语义令牌中...")
    code_task = llm_model(code_task)
    semantic_token = code_task.result.reshape(-1)

    # 将LLM移回CPU，释放GPU内存
    llm_model = llm_model.cpu()
    torch.cuda.empty_cache()

    # 将扩散模型移至GPU
    diffusion_model = diffusion_model.cuda()
    semantic_token = semantic_token.cuda()

    # 使用语义令牌生成视频
    video_path = f"{save_path}.mp4"
    video_task = VideoTask(
        save_file_name=video_path,
        prompt=prompt,
        seed=seed,
        fps=8,
        semantic_token=semantic_token,
    )

    print("🎥 扩散模型生成视频中...")
    video_task = diffusion_model(video_task)
    video = video_task.result

    # 保存视频
    save_video_tensor(video, video_path, fps=video_task.fps)
    print(f"✅ 视频生成完成，保存至: {video_path}")

    # 将扩散模型移回CPU，释放GPU内存
    diffusion_model = diffusion_model.cpu()
    torch.cuda.empty_cache()

    return video_path


# Gradio界面
def create_ui(llm_model, diffusion_model):
    with gr.Blocks(title="LanDiff: 文本到视频生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # LanDiff: 集成语言模型和扩散模型的视频生成
            
            提供一段详细的文本描述，LanDiff将为您生成相应的视频。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="文本提示",
                    placeholder="请输入详细的场景描述，例如：一只棕色和棕褐色壳的蜗牛在一张绿色的苔藓床上爬行。蜗牛的身体是灰褐色的，有两个突出的触角向前伸展。环境表明这是一个自然的户外环境，重点是蜗牛在苔藓表面上的移动。",
                    lines=5,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        cfg_scale = gr.Slider(
                            label="CFG尺度",
                            minimum=1.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            info="控制文本提示对生成内容的影响程度",
                        )

                    with gr.Column(scale=1):
                        motion_score = gr.Slider(
                            label="运动分数",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            info="控制视频中运动的程度",
                        )

                with gr.Row():
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                        info="设置随机种子以获得可重复的结果",
                    )
                    random_seed_btn = gr.Button("随机种子", size="sm")

                generate_btn = gr.Button("生成视频", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="生成的视频")
                status = gr.Markdown("等待生成...")

        # 处理函数
        def process(prompt, cfg, motion, seed_value):
            try:
                # yield "🔄 正在生成视频，请稍候...", None

                video_path = generate_video(
                    prompt, cfg, motion, llm_model, diffusion_model, int(seed_value)
                )
                print("✅ 视频生成成功！")
                return "✅ 视频生成成功！", video_path
            except Exception as e:
                print(f"❌ 生成失败: {str(e)}")
                return f"❌ 生成失败: {str(e)}", None

        def get_random_seed():
            import random

            return int(random.randint(0, 2**31 - 1))

        random_seed_btn.click(fn=get_random_seed, inputs=[], outputs=[seed])

        generate_btn.click(
            fn=process,
            inputs=[prompt, cfg_scale, motion_score, seed],
            outputs=[status, video_output],
        )

        gr.Markdown(
            """
            ### 示例提示:
            1. 一只棕色和棕褐色壳的蜗牛在一张绿色的苔藓床上爬行。蜗牛的身体是灰褐色的，有两个突出的触角向前伸展。
            2. 一片宁静的海滩，金色的沙子上卷起波浪，海水是清澈的蓝绿色，与远处的蓝天相融合。几片白云飘在天空中。
            3. 在满是积雪的深山中，一条偏远的火车轨道蜿蜒穿过。一列红色的火车冒着蒸汽，缓缓前行，在雪地上留下轨迹。
            """
        )

        gr.Markdown(
            """
            ---
            ### 关于LanDiff
            LanDiff是一个集成语言模型和扩散模型的视频生成系统，能够从详细的文本描述生成高质量视频内容。
            
            [项目主页](https://landiff.github.io/) | [GitHub](https://github.com/LanDiff/LanDiff) | [论文](https://arxiv.org/abs/2503.04606)
            """
        )

    return demo


# 主函数
def main():
    # 模型路径设置
    llm_ckpt_path = "ckpts/LanDiff/llm/model.safetensors"
    diffusion_ckpt_path = "ckpts/LanDiff/diffusion"

    print("🔧 正在设置环境...")
    # 初始化模型（先在CPU上）
    llm_model, diffusion_model = initialize_models(llm_ckpt_path, diffusion_ckpt_path)

    # 创建和启动Gradio接口
    demo = create_ui(llm_model, diffusion_model)
    demo.queue(max_size=10).launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
