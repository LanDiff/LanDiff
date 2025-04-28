import copy

import fiddle as fdl
import torch
from fiddle import Config
from torch import nn

from landiff.llm.models.lm_model import Semantic1DLM
from landiff.llm.models.transformer import GPT
from landiff.llm.modules.conditioner import MicroConditioner, TextCond
from landiff.llm.modules.text_encoder import FlanT5XXL
from landiff.llm.modules.tokenizer import SemanticFrozenTokenizer
from landiff.llm.modules.transformer_blocks import LlamaTransformerBlock
from landiff.modules.pos_emb import Rope1DPosEmb
from landiff.tokenizer.tokenizer_cfg import build_tokenizer


def build_llm() -> Config:
    """Builds the LLM model configuration."""
    tokenizer_cfg = Config(
        SemanticFrozenTokenizer,
        tokenizer=build_tokenizer(),
        ckpt_path="ckpts/LanDiff/tokenizer/model.safetensors",
    )
    transformer_block_cfg = Config(
        LlamaTransformerBlock,
        num_heads=16,
        hidden_dim=2048,
        mlp_dim=11008,
        activation=nn.GELU(approximate="tanh"),
        drop_path=0.0,
    )
    rope_cfg = Config(
        Rope1DPosEmb,
        dim=128,
        theta_base=10000,
        max_len=32768,
    )
    transformer_cfg = Config(
        GPT,
        tokenizer_cfg.tokenizer.quantizer.codebook_size + 7,
        hidden_dim=2048,
        causal=True,
        fwd_dtype=torch.bfloat16,
        blocks=[copy.deepcopy(transformer_block_cfg) for _ in range(24)],
        rope=rope_cfg,
    )
    cfg = Config(
        Semantic1DLM,
        train2d=False,
        train3d=True,
        Iframe_len=330,
        Pframe_len=74,
        predict_motion_score=False,
        caculate_motion_socre_loss=False,
        fwd_dtype=torch.bfloat16,
        tokenizer=tokenizer_cfg,
        transformer=transformer_cfg,
        cond_model=Config(
            TextCond,
            text_encoder=Config(FlanT5XXL, load_weights=True, fwd_dtype=torch.bfloat16),
            max_cond_tokens_num=512,  # the max number of tokens for flan-t5 is 512, for CLIP is 77
            embed_dim=2048,
            padding=False,
            freeze_text_encoder=True,
            cfg_drop_prob=0.1,
            use_mlp_embeddings=True,
        ),
        micro_condition=Config(
            MicroConditioner,
            out_dim=2048,
            frequency_embedding_size=256,
            crossattn_condition_keys=(
                "frames",
                "motion_score",
            ),
            fwd_dtype=torch.bfloat16,
            defaults={"frames": 1, "motion_score": 0},
        ),
    )
    return cfg


if __name__ == "__main__":
    # Build the tokenizer model
    model_cfg = build_llm()
    model = fdl.build(model_cfg)
    print(model)
