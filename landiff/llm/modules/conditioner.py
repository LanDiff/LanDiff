import logging
from copy import copy
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat

from landiff.diffusion.sgm.modules.diffusionmodules.util import timestep_embedding
from landiff.utils import freeze_model, maybe_autocast

from .transformer_blocks import MLP2

logger = logging.getLogger(__name__)


class MicroConditioner(nn.Module):
    """Embeds condition scalar, e.g. image size, aspect ratio, into vector representations."""

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 512,
        frequency_embedding_size: int = 256,
        adaln_condition_keys: tuple[str, ...] = (),
        crossattn_condition_keys: tuple[str, ...] = (),
        defaults: dict[str, float | int] | None = None,
        adaln_condition_pooling_mode: str = "mean",
        fwd_dtype=torch.float32,
        drop_probs: dict[str, float] = {},
    ):
        """Microcondition will be embedded to `frequency_embedding_size` and then projected to hidden_dim and out_dim with a 2-layer MLP.

        Args:
            adaln_condition_keys, crossattn_condition_keys: tuple of condition keys to be used for adaln and crossattn respectively.
            defaults: dict of default values for the keys, if not found in data.
            adaln_condition_pooling_mode: can be 'sum', 'mean' or 'mlp', use a mlp to pool to adaln_condition from (B, N, D) to (B, D),
                or 'sum' or 'mean' along the 'N' dimension. TODO 'mlp' mode will be remove in the future.
            drop_probs: dict of drop probabilities in [0, 1] for each condition key, 0 for no drop, 1 for always drop.
        """
        super().__init__()
        assert (
            len(adaln_condition_keys) + len(crossattn_condition_keys) > 0
        ), "No condition keys provided"
        assert adaln_condition_pooling_mode in ["sum", "mean", "mlp"]
        # Sort is important to avoid inconsistent order between GPUs
        all_condition_keys = sorted(
            list(set(adaln_condition_keys + crossattn_condition_keys))
        )
        self.mlps = nn.ModuleDict()
        for condition_key in all_condition_keys:
            self.mlps[condition_key] = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim, bias=True),
            )
        # logger.info(f'Using micro condition: {all_condition_keys}')

        self.frequency_embedding_size = frequency_embedding_size
        self.all_condition_keys = all_condition_keys
        self.adaln_condition_keys = adaln_condition_keys
        self.crossattn_condition_keys = crossattn_condition_keys
        self.defaults = copy(defaults or {})
        self.adaln_condition_pooling_mode = adaln_condition_pooling_mode
        self.fwd_dtype = fwd_dtype
        self.drop_probs = drop_probs

        self.null_cond_embeddings = nn.ParameterDict()
        for cond_key in sorted(list(self.drop_probs.keys())):
            assert (
                cond_key in all_condition_keys
            ), f"Drop condition key {cond_key} not found in all_condition_keys"
            drop_prob = self.drop_probs[cond_key]
            if drop_prob > 0:
                # logger.info(f'Using micro condition with drop prob for training: {drop_prob} for condition: {cond_key}')
                self.null_cond_embeddings[cond_key] = nn.Parameter(
                    torch.randn(frequency_embedding_size)
                    / frequency_embedding_size**0.5
                )

        if adaln_condition_pooling_mode == "mlp":
            self.adaln_out = nn.Sequential(
                nn.SiLU(),
                nn.Linear(len(adaln_condition_keys) * out_dim, out_dim, bias=True),
            )

        self.reset_parameters()

    @property
    def device(self):
        return next(self.mlps.parameters()).device

    def reset_parameters(self):
        for _, mlp in self.mlps.items():
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)
        if self.adaln_condition_pooling_mode == "mlp":
            nn.init.zeros_(self.adaln_out[1].weight)  # same as above
            nn.init.zeros_(self.adaln_out[1].bias)

    def forward(
        self, x: dict[str, Any], rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            x: batched inputs.

        Note about drop condition:
            train: conditions are dropped w.r.t self.drop_probs.
            eval: set x[cond_key] = None to drop condition. If cond_key is not presented in x, will use default condition instead of dropping.

        Returns:
            Adaln condition tensor of shape (B, out_dim).
            Crossattn condition tensor of shape (B, num_conditions, out_dim).
        """
        main_input = x.get("class_id", x.get("caption", x.get("image", x.get("video"))))
        batch_size = len(main_input)
        adaln_conditions, crossattn_conditions = [], []
        for cond_key in self.all_condition_keys:
            if self.training:
                # TODO: should assert cond_key in x?
                cond = x.get(cond_key, None)  # (B,)
            else:
                cond = x.get(cond_key, self.defaults.get(cond_key))

            if cond is None:
                if cond_key in self.drop_probs and self.drop_probs[cond_key] > 0:
                    emb = (
                        self.null_cond_embeddings[cond_key]
                        .expand(batch_size, -1)
                        .to(self.fwd_dtype)
                    )
                else:
                    raise ValueError(
                        f"Condition key {cond_key} not found in data, and a default is not given, and null default is not set."
                    )
            else:
                if isinstance(cond, (int, float)):
                    cond = torch.full((batch_size,), cond, device=self.device)
                else:
                    assert isinstance(
                        cond, torch.Tensor
                    ), f"cond key: {cond_key}, type: {type(cond)}"
                    assert cond.shape == (
                        batch_size,
                    ), f"cond key: {cond_key}, shape: {cond.shape}"
                emb = timestep_embedding(cond, self.frequency_embedding_size).to(
                    self.fwd_dtype
                )  # (B, frequency_embedding_size)
                if (
                    cond_key in self.drop_probs
                    and self.drop_probs[cond_key] > 0
                    and self.training
                ):
                    null_emb = (
                        self.null_cond_embeddings[cond_key]
                        .expand(batch_size, -1)
                        .to(self.fwd_dtype)
                    )
                    keep_mask = (
                        torch.zeros((batch_size, 1), device=self.device).uniform_(
                            0, 1, generator=rng
                        )
                        > self.drop_probs[cond_key]
                    )
                    emb = torch.where(keep_mask, emb, null_emb)

            with maybe_autocast(emb, torch.bfloat16):
                emb = self.mlps[cond_key](emb)  # (B, out_dim)
            if cond_key in self.adaln_condition_keys:
                adaln_conditions.append(emb)
            if cond_key in self.crossattn_condition_keys:
                crossattn_conditions.append(emb)
        adaln_conditions = (
            torch.stack(adaln_conditions, dim=1) if len(adaln_conditions) > 0 else None
        )  # (B, num_cond_adaln, out_dim)
        crossattn_conditions = (
            torch.stack(crossattn_conditions, dim=1)
            if len(crossattn_conditions) > 0
            else None
        )  # (B, num_cond_crossattn, out_dim)

        if adaln_conditions is not None:
            if self.adaln_condition_pooling_mode == "mean":
                adaln_conditions = torch.mean(adaln_conditions, dim=1)  # (B, out_dim)
            elif self.adaln_condition_pooling_mode == "sum":
                adaln_conditions = torch.sum(adaln_conditions, dim=1)  # (B, out_dim)
            else:  # 'mlp'
                with maybe_autocast(adaln_conditions, torch.bfloat16):
                    adaln_conditions = self.adaln_out(
                        rearrange(adaln_conditions, "b n d -> b (n d)")
                    )  # (B, out_dim)
        return adaln_conditions, crossattn_conditions


class TextCond(nn.Module):
    """Text conditioning module.

    Wrap an existing text encoder with CFG logic,
    and add an extra projection (which has the same fwd dtype with given text_encoder.)
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        max_cond_tokens_num: int,
        embed_dim: int,
        padding: bool,
        freeze_text_encoder: bool = True,
        cfg_drop_prob: float = 0.0,
        use_mlp_embeddings: bool = False,
    ):
        """
        Args:
          text_encoder: A text encoder defined in nn/text_encoder.py
          max_cond_tokens_num: Maximum number of conditional tokens. We use this instead of using tokenizer.model_max_length because sometimes tokenizer.model_max_length by default is equal to 1000000000000000019884624838656 in huggingface, which is too large.
          embed_dim: Embedded dimension of the conditional tokens.
          padding: whether to left-pad texts in a batch to the maximum length. If True, forward() will return a tensor of batched encoding.
            If False, forward() will return a list of encoded texts in different lengths.
          freeze_encoder: Freeze the text encoder or not, default is True.
          cfg_drop_prob: classifier-free guidance drop probability.
          use_mlp_embeddings: use a mlp block as the learnable projection layers, like PixArt.
        """
        super().__init__()

        self.max_cond_tokens_num = max_cond_tokens_num
        assert isinstance(padding, bool), padding
        self.padding = padding
        self.freeze_text_encoder = freeze_text_encoder
        self.cfg_drop_prob = cfg_drop_prob
        self.text_encoder = text_encoder
        self.use_mlp_embeddings = use_mlp_embeddings
        self.embed_dim = embed_dim
        if use_mlp_embeddings:
            self.embeddings = MLP2(
                dims=[self.text_encoder.dimension, embed_dim, embed_dim],
                activation=nn.GELU(approximate="tanh"),
            )
        else:
            self.embeddings = nn.Linear(self.text_encoder.dimension, embed_dim)
        self.fwd_dtype = text_encoder.fwd_dtype

        tokenizer_max_token_length = self.text_encoder.max_length
        if tokenizer_max_token_length < self.max_cond_tokens_num:
            logger.warning(
                f"The maximum conditional token number ({self.max_cond_tokens_num}) is LARGER "
                f"than the maximum token length of the tokenizer {type(self.text_encoder)} ({tokenizer_max_token_length})"
            )
            logger.warning(
                f"Use the tokenizer's maximum token length ({tokenizer_max_token_length}) "
                "as the maximum conditional token number."
            )
            self.max_cond_tokens_num = tokenizer_max_token_length

        if self.freeze_text_encoder:
            logger.info(f"Freeze text encoder: {type(text_encoder)}")
            freeze_model(self.text_encoder)

        # classifier-free guidance
        if self.cfg_drop_prob > 0:
            self.null_text_embedding = nn.Parameter(
                torch.randn(embed_dim) / embed_dim**0.5
            )

    @property
    def device(self):
        return next(self.embeddings.parameters()).device

    def forward(
        self, x: list[str], rng: torch.Generator | None = None
    ) -> list[torch.FloatTensor] | torch.FloatTensor:
        """Encode a list of texts.

        Returns a list of features, each has shape (len_i, self.embed_dim)
        """
        text_embedding, text_mask = self.text_encoder.encode_texts_padded(
            x
        )  # text_mask: 1 means valid.
        with maybe_autocast(text_embedding):
            text_embedding = self.embeddings(text_embedding)
            assert (
                text_embedding.dtype == self.fwd_dtype
            ), f"output dtype: {text_embedding.dtype}, fwd_dtype: {self.fwd_dtype}"

        if self.cfg_drop_prob > 0 and self.training:
            batch_size, tokens_num, _ = text_embedding.shape
            keep_mask = (
                torch.zeros((len(x), 1, 1), device=self.device).uniform_(
                    0, 1, generator=rng
                )
                > self.cfg_drop_prob
            )
            null_text_embedding = repeat(
                self.null_text_embedding, "d -> b n d", b=batch_size, n=tokens_num
            ).to(dtype=text_embedding.dtype)
            text_embedding = torch.where(
                keep_mask, text_embedding, null_text_embedding
            )  # use null text embedding to mask out text embedding

        if self.padding:
            return text_embedding
        else:
            return [text_embedding[i, text_mask[i]] for i in range(len(x))]

    def _drop_token_embedding(
        self, embedding, rng, null_embed=None, cfg_drop_prob=None
    ):
        null_embed = self.null_text_embedding if null_embed is None else null_embed
        cfg_drop_prob = self.cfg_drop_prob if cfg_drop_prob is None else cfg_drop_prob
        keep_mask = (
            torch.zeros(len(embedding), device=self.device).uniform_(
                0, 1, generator=rng
            )
            > cfg_drop_prob
        )
        null_embed = (
            null_embed.reshape(1, -1).to(dtype=self.fwd_dtype)
            if cfg_drop_prob > 0
            else None
        )
        return [
            torch.where(keep_mask[idx], embed, null_embed)
            for idx, embed in enumerate(embedding)
        ]

    def forward_with_precomputed_embedding(
        self, embedding: list[torch.FloatTensor], rng: torch.Generator | None = None
    ) -> list[torch.FloatTensor]:
        assert (
            not self.padding
        ), "Precomputed embedding is not supported when padding is True."
        assert isinstance(embedding, list), type(embedding)
        assert embedding[0].ndim == 2, embedding[0].shape
        ret = []
        for embed in embedding:
            embed = embed.to(self.fwd_dtype)
            with maybe_autocast(embed, torch.bfloat16):
                embed = self.embeddings(embed)
                assert (
                    embed.dtype == self.fwd_dtype
                ), f"output dtype: {embed.dtype}, fwd_dtype: {self.fwd_dtype}"
                assert embed.ndim == 2, embed.shape
            ret.append(embed)
        if self.training and self.cfg_drop_prob > 0:
            ret = self._drop_token_embedding(ret, rng)
        return ret

    def forward_unconditional(
        self, x: list[str]
    ) -> list[torch.FloatTensor] | torch.FloatTensor:
        tokenized_text = self.text_encoder.tokenize_padded(x)
        text_embedding = repeat(
            self.null_text_embedding,
            "d -> b n d",
            b=len(x),
            n=tokenized_text.input_ids.shape[1],
        ).to(dtype=self.fwd_dtype)
        if self.padding:
            return text_embedding
        else:
            text_mask = tokenized_text.attention_mask.bool()
            return [text_embedding[i, text_mask[i]] for i in range(len(x))]
