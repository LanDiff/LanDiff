import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

from landiff.utils import freeze_model


class SemanticEmbeddingModel(nn.Module):
    """Visual embedding for fixed-length visual tokens."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # input token and position embedding
        self.tok_emb_code = nn.Embedding(self.vocab_size, self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (..., H, W) tokens

        Returns:
            (..., H, W, embed_dim) features
        """
        assert not x.dtype.is_floating_point, f"x.dtype: {x.dtype}"
        x = self.tok_emb_code(x)
        return x

    def lookup(self, x: list[int] | torch.LongTensor):
        """Lookup embeddings for some indices.

        Returns:
            (len(x), embed_dim)
        """
        if not torch.is_tensor(x):
            if len(x) == 1:  # skip Tensor construction
                assert (
                    self.tok_emb_code.weight.shape[0] == self.vocab_size
                )  # TODO @keyu: ensure the weight not sharded by FSDP; TODO once confirmed, remove this
                return self.tok_emb_code.weight[x[0]].unsqueeze(0)
            else:
                x = torch.tensor(x, device=self.tok_emb_code.weight.device)

        assert x.ndim == 1
        return self.tok_emb_code(x)


class SemanticFrozenTokenizer(nn.Module):

    def __init__(
        self,
        tokenizer: nn.Module,
        ckpt_path: None | str = None,
        segment_length: int = 13,
        segment_stride: int = 13,
    ):
        super().__init__()
        # load tokenizer
        from landiff.tokenizer.models.video_titok_vq import VideoVQ

        self.tokenizer: VideoVQ = tokenizer
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        if ckpt_path is not None:
            assert os.path.exists(
                ckpt_path
            ), f"Please provide the correct path of the weight of the tokenizer, the path you provide is: {ckpt_path}"
            if ckpt_path.endswith("safetensors"):
                state_dict = load_file(ckpt_path)
                self.tokenizer.load_state_dict(state_dict, strict=True)
            else:
                raise ValueError(
                    f"Only safetensors is supported, but the file you provide is: {ckpt_path}"
                )
        freeze_model(self)

    def vocab_size(self):
        return self.tokenizer.quantizer.codebook_size

    def encode_codes(self, visual: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Return flattened codes of the given visual."""
        # visual shape b,t,...
        if self.segment_length != self.segment_stride:
            endoced_codes = []
            for offset in range(0, visual.shape[1], self.segment_stride):
                segment = visual[:, offset : offset + self.segment_length]
                _, indices = self.tokenizer.encode_to_index(segment, **kwargs)
                endoced_codes.append(indices)
            return endoced_codes
        else:
            _, indices = self.tokenizer.encode_to_index(visual, **kwargs)
            return indices
