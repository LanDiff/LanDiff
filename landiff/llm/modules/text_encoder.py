import logging

import torch
from torch import nn
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

logger = logging.getLogger(__name__)


class BaseTextEncoder(nn.Module):
    """Base text encoder."""

    def __init__(
        self,
        Tokenizer: PreTrainedTokenizer,
        EncoderModel: PreTrainedModel,
        model_path: str,
        max_length: int,
        load_weights: bool = True,
        fwd_dtype=torch.bfloat16,
    ):
        """
        Args:
            Tokenizer: tokenizer class.
            EncoderModel: text encoder class.
            model_path: path to the model checkpoints.
            max_length: max length of tokenized text.
            load_weights: whether to load weights. False can be used with precomputed embedding during training.
            fwd_dtype: tensor dtype.
        """
        super().__init__()
        self.tokenizer = Tokenizer.from_pretrained(model_path, padding_side="left")
        if load_weights:
            self.text_encoder = EncoderModel.from_pretrained(
                model_path, torch_dtype=fwd_dtype
            )
            logger.info("Loaded %s model from %s", type(self).__name__, model_path)
            self._hf_config = self.text_encoder.config
        else:
            logger.info("Not loading model for %s", type(self).__name__)
            self.text_encoder = None
            self._hf_config = AutoConfig.from_pretrained(model_path)

        try:
            cfg_dtype = self._hf_config.torch_dtype
        except Exception as e:
            logger.error(
                f"get torch_dtype error: {self._hf_config}, Exception: " + str(e)
            )
        else:
            if cfg_dtype is not None and cfg_dtype != fwd_dtype:
                logger.warning(
                    f"Text encoder {type(self).__name__} uses {cfg_dtype} in its huggingface config."
                    f" But we're using {fwd_dtype}"
                )
        self.max_length = max_length
        self.fwd_dtype = fwd_dtype

    @property
    def dimension(self) -> int:
        """Dimension of model."""
        return self._hf_config.d_model

    @property
    def device(self):
        """Get current device."""
        return next(self.parameters()).device

    def tokenize_padded(self, texts: list[str]):
        """Tokenize text with truncation.

        Returns:
            tokenized text and attention mask.
        """
        assert isinstance(texts, list) and not isinstance(texts, str), texts
        tokenized_text = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_length,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            return_attention_mask=True,
        )
        # NOTE that the returned attn_mask is not boolean.
        # Can we fix it?
        return tokenized_text

    def encode_texts_padded(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the given texts and return padded features.

        Returns:
            batch x max_len x dim features.
            batch x max_len padding mask, False means padding.
        """
        for txt in texts:
            assert isinstance(txt, str), txt
        assert self.text_encoder is not None, "Model not loaded"
        tokenized_text = self.tokenize_padded(texts)
        input_ids = tokenized_text.input_ids.to(device=self.device)
        attn_mask = tokenized_text.attention_mask.to(device=self.device)
        output = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_embedding = output.last_hidden_state  # (batch_size, num_tokens, embed_dim)
        assert text_embedding.dtype == self.fwd_dtype, text_embedding.dtype
        return text_embedding, attn_mask.bool()

    def encode_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Encode the given texts and return unpadded features.

        Returns:
            a list of features, each has shape seqlen x dim.
        """
        embedding, attn_mask = self.encode_texts_padded(texts)
        embedding = [embedding[i][attn_mask[i]] for i in range(embedding.shape[0])]
        return embedding


class T5TextEncoder(BaseTextEncoder):
    """T5 Text Encoder."""

    def __init__(self, *args, **kwargs):
        """
        Args:
            model_path: path to the model checkpoints.
            load_weights: whether to load weights. False can be used with precomputed embedding during training.
        """
        super().__init__(T5Tokenizer, T5EncoderModel, *args, **kwargs)


class FlanT5XXL(T5TextEncoder):

    def __init__(
        self,
        model_path: str = "google/flan-t5-xxl",
        max_length: int = 512,
        load_weights: bool = True,
        fwd_dtype=torch.bfloat16,
    ):
        super().__init__(model_path, max_length, load_weights, fwd_dtype)
