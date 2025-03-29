import fiddle as fdl
import torch
import vector_quantize_pytorch
from fiddle import Config

from landiff.modules.pos_emb import Rope3DPosEmb
from landiff.tokenizer.models.feature_extractor.theia_extractor import TheiaExtractor
from landiff.tokenizer.models.video_titok_vq import VideoVQ
from landiff.tokenizer.modules.blocks import (
    AttentionImp,
    AttentionMaskType,
    PositionalEmbedingType,
    TiTokDecoder,
    TiTokEncoder,
)


def build_feature_extractor() -> Config:
    cfg = Config(
        TheiaExtractor,
        micro_batch_size=1,
        interpolate=True,
        output_shape=(30, 45),
        bfp16=True,
    )
    return cfg


def build_tokenizer() -> Config:
    rope_layer_cfg = Config(
        Rope3DPosEmb,
        dim=64,
        max_time=100,
        max_height=30,
        max_width=45,
        one_dim_max_time=100000,
        multiple=16,
        device="cpu",
    )
    encoder_cfg = Config(
        TiTokEncoder,
        image_size=(30, 45),
        image_channels=768,
        patch_size=1,
        model_size="base",
        num_latent_tokens=1218,  # downsample 4x
        token_size=768,
        use_checkpoint=False,
        qk_norm=False,
        causal=False,
        bias=False,
        width=768,
        num_layers=12,
        num_heads=12,
        positional_embedding_type=PositionalEmbedingType.ROPE_3D,
        rope_layer=rope_layer_cfg,
        attention_imp=AttentionImp.FLEX_ATTENTION,
        attention_mask_type=AttentionMaskType.VIDEO_ENCODER_MASK,
        use_cls_token=False,
        temporal_size=13,
        PFrame_tokens=74,
        inside_latent_tokens=True,
    )
    decoder_cfg = Config(
        TiTokDecoder,
        image_size=encoder_cfg.image_size,
        image_channels=encoder_cfg.image_channels,
        patch_size=encoder_cfg.patch_size,
        model_size=encoder_cfg.model_size,
        width=encoder_cfg.width,
        num_layers=encoder_cfg.num_layers,
        num_heads=encoder_cfg.num_heads,
        num_latent_tokens=encoder_cfg.num_latent_tokens,
        token_size=encoder_cfg.token_size,
        output_channels=encoder_cfg.image_channels,
        use_checkpoint=False,
        qk_norm=False,
        bias=False,
        causal=False,
        code_drop=False,
        positional_embedding_type=PositionalEmbedingType.ROPE_3D,
        rope_layer=rope_layer_cfg,
        attention_imp=AttentionImp.FLEX_ATTENTION,
        attention_mask_type=AttentionMaskType.VIDEO_DECODER_MASK,
        use_cls_token=encoder_cfg.use_cls_token,
        temporal_size=encoder_cfg.temporal_size,
        PFrame_tokens=encoder_cfg.PFrame_tokens,
    )
    quantizer_cfg = vector_quantize_pytorch.VectorQuantize(
        codebook_size=2048,
        dim=encoder_cfg.token_size,
        kmeans_init=True,
        threshold_ema_dead_code=2,
        codebook_dim=16,
    )
    cfg = Config(
        VideoVQ,
        feature_extractor=build_feature_extractor(),
        fwd_dtype=torch.bfloat16,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        num_latent_tokens=None,
        quantizer=quantizer_cfg,
        re_loss_fn=torch.nn.MSELoss(reduction="mean"),
        commit_loss_weight=1.0,
        model_type="transformer",
        mean_std_dim=768,
        ckpt_path="ckpts/LanDiff/tokenizer/model.safetensors",
    )

    return cfg


if __name__ == "__main__":
    # Build the tokenizer model
    model_cfg = build_tokenizer()
    model = fdl.build(model_cfg)
    print(model)
