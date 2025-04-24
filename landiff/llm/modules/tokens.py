from __future__ import annotations

import torch
from torch import nn


class Vocab:
    def __init__(self):
        self._range: dict[str, tuple[int, int]] = {}
        self._specials: dict[str, int] = {}
        self._size = 0

    def size(self):
        return self._size

    def add_special(self, name: str):
        self._specials[name] = self._size
        self._size += 1

    def add_range(self, name: str, size: int):
        assert isinstance(size, int) and size > 0, size
        self._range[name] = (self._size, self._size + size)
        self._size += size

    def __getattr__(self, name):
        return self._specials[name]


class TokenSequence:
    """A class for building and managing token sequences, supporting storage and operations of tokens, features, and loss masks."""

    IGNORE_INDEX = -100
    # Use -100 because it happens to be the default in F.cross_entropy

    def __init__(self):
        self._tokens = []
        self._features = []
        self._loss_mask = []
        self._ranges = {}  # More descriptive naming
        self._length = 0

    def __len__(self):
        return self._length

    @property
    def device(self):
        """Return the device where the current sequence is located"""
        if not self._tokens:
            raise ValueError("Empty sequence has no device")
        return self._tokens[0].device

    def append_token(
        self,
        tokens: torch.LongTensor,
        loss_mask: bool | torch.BoolTensor,
        *,
        name: str | None = None,
    ):
        """Add tokens to the sequence"""
        start = len(self)
        assert tokens.ndim == 1, f"Expected 1D tensor, got {tokens.ndim}D"
        assert tokens.dtype in [
            torch.long,
            torch.int32,
        ], f"Invalid dtype: {tokens.dtype}"
        assert isinstance(
            loss_mask, (bool, torch.Tensor)
        ), f"Invalid loss_mask type: {type(loss_mask)}"

        if isinstance(loss_mask, torch.Tensor):
            assert (
                loss_mask.dtype == torch.bool
            ), f"Expected bool tensor, got {loss_mask.dtype}"
            assert len(loss_mask) == len(
                tokens
            ), f"Mask length {len(loss_mask)} != tokens length {len(tokens)}"
        else:
            loss_mask = torch.full(
                (len(tokens),),
                fill_value=loss_mask,
                dtype=torch.bool,
                device=tokens.device,
            )

        self._tokens.append(tokens.to(torch.long))
        self._loss_mask.append(loss_mask)
        self._features.append(None)
        self._length += len(tokens)

        if name:
            assert name not in self._ranges, f"Range name '{name}' already exists"
            self._ranges[name] = (start, len(self))

    def append_feature(self, feature: torch.FloatTensor, *, name: str | None = None):
        """Add features to the sequence"""
        start = len(self)
        assert feature.ndim == 2, f"Expected 2D feature, got shape {feature.shape}"

        self._features.append(feature)
        self._loss_mask.append(
            torch.zeros(
                size=(feature.shape[0],), dtype=torch.bool, device=feature.device
            )
        )
        self._tokens.append(
            torch.full(
                (feature.shape[0],),
                fill_value=self.IGNORE_INDEX,
                dtype=torch.long,
                device=feature.device,
            )
        )
        self._length += feature.shape[0]

        if name:
            assert name not in self._ranges, f"Range name '{name}' already exists"
            self._ranges[name] = (start, len(self))

    def append_token_and_feature(
        self,
        token: torch.LongTensor,
        feature: None | torch.FloatTensor,
        loss_mask: bool | torch.BoolTensor,
        *,
        name: str | None = None,
    ):
        """Add both token and corresponding feature simultaneously"""
        self.append_token(token, loss_mask, name=name)
        if feature is not None:
            assert len(feature) == len(
                token
            ), f"Feature length {len(feature)} != token length {len(token)}"
            assert feature.ndim == 2, f"Expected 2D feature, got {feature.ndim}D"
            self._features[-1] = feature

    def get_tokens(self) -> torch.LongTensor:
        """Get all tokens after concatenation"""
        return torch.cat(self._tokens, dim=0)

    def get_loss_mask(self) -> torch.BoolTensor:
        """Get all loss masks after concatenation"""
        return torch.cat(self._loss_mask, dim=0)

    def get_features(self, embedding: nn.Embedding | None = None) -> torch.FloatTensor:
        """Get all features after concatenation, use embedding to get features if feature is None"""
        features = []
        for tokens, f in zip(self._tokens, self._features):
            if f is None:
                assert (
                    embedding is not None
                ), "Embedding must be provided for tokens without features"
                f = embedding(tokens)
            features.append(f)
        return torch.cat(features, dim=0)

    @staticmethod
    def collate(sequences: list[TokenSequence]):
        """Combine multiple sequences into batch tensors"""
        lengths = [len(seq) for seq in sequences]
        assert (
            len(set(lengths)) == 1
        ), f"All sequences must have the same length, got {lengths}"

        tokens = torch.stack([s.get_tokens() for s in sequences], dim=0)  # [N, seqlen]
        features = torch.stack(
            [s.get_features(None) for s in sequences], dim=0
        )  # [N, seqlen, dim]
        loss_mask = torch.stack(
            [s.get_loss_mask() for s in sequences], dim=0
        )  # [N, seqlen]
        return tokens, features, loss_mask
