from __future__ import annotations

import torch
from torch import nn


class Vocab:
    """
    Vocabulary for normal (e.g. visual) and sepcial tokens.
    Use self.add_range to set the range for normal tokens.
    Use self.add_special to add a sepcial token with its name and id.

    Example:
        >>> v = Vocab()
        >>> v.add_range("visual", 8192)
        >>> print(v._range)
        {"visual": (0, 8192)}  # left-inclusive, right-exclusive
        >>> print(v._size)
        8192
        >>> v.add_special("BOS")
        >>> v.add_special("START_OF_IMG")
        >>> v.add_special("NEWLINE")
        >>> print(v._specials)
        {"BOS": 8192, "START_OF_IMG": 8193, "NEWLINE": 8194}
        >>> print(v._size)
        8195
        >>> print(v.START_OF_IMG)
        8193
    """

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


class SequenceBuilder:
    """
    A helper class to construct a sequence of tokens + features + loss_mask.
    """

    IGNORE_INDEX = -100
    # Use -100 because it happens to be the default in F.cross_entropy

    def __init__(self):
        self._tokens = []  # list of 1D int64 token ids.
        self._features = []  # list of 2D float features.
        self._loss_mask = []  # list of 1D bool masks.
        self._name_to_range = {}
        self._length = 0

    def get_range(self, name) -> tuple[int, int]:
        """Returns the start(inclusive),end(exclusive) index of "name" in the sequence."""
        return self._name_to_range[name]

    def __len__(self):
        return self._length

    @property
    def device(self):
        return self._tokens[0].device

    def add_token(
        self,
        tokens: torch.LongTensor,
        loss_mask: bool | torch.BoolTensor,
        *,
        name: str | None = None,
    ):
        """
        Append a sequence of tokens.
        Args:
            loss_mask: whether to compute loss on the token. True means compute loss.
            name: an optional name to be queried later
        """
        start = len(self)
        assert tokens.ndim == 1
        assert tokens.dtype in [torch.long, torch.int32], tokens.dtype
        assert isinstance(loss_mask, (bool, torch.Tensor))

        if isinstance(loss_mask, torch.Tensor):
            assert loss_mask.dtype == torch.bool
            assert len(loss_mask) == len(tokens)
        else:
            loss_mask = tokens.new_empty(len(tokens), dtype=torch.bool).fill_(loss_mask)

        self._tokens.append(tokens.to(torch.long))
        self._loss_mask.append(loss_mask)
        self._features.append(None)
        self._length += len(tokens)
        if name:
            assert name not in self._name_to_range, name
            self._name_to_range[name] = (start, len(self))

    def add_feature(self, feature: torch.FloatTensor, *, name: str | None = None):
        """Append pre-computed features to the sequence."""
        start = len(self)
        assert feature.ndim == 2, feature.shape
        self._features.append(feature)
        # No loss on precomputed features. Use IGNORE_INDEX as token placeholder.
        self._loss_mask.append(
            feature.new_zeros(size=(feature.shape[0],), dtype=torch.bool)
        )
        self._tokens.append(
            feature.new_empty(feature.shape[0], dtype=torch.long).fill_(
                self.IGNORE_INDEX
            )
        )
        self._length += feature.shape[0]
        if name:
            assert name not in self._name_to_range, name
            self._name_to_range[name] = (start, len(self))

    def add_token_and_feature(
        self,
        token: torch.LongTensor,
        feature: None | torch.FloatTensor,
        loss_mask: bool | torch.BoolTensor,
        *,
        name: str | None = None,
    ):
        """
        Append a sequence of token together with its features.
        If feature is None, this is equivalent to `add_token`.
        """
        self.add_token(token, loss_mask, name=name)
        if feature is not None:
            assert len(feature) == len(token)
            assert feature.ndim == 2
            self._features[-1] = feature

    def concat_tokens(self) -> torch.LongTensor:
        """Returns a vector of tokens. Use "IGNORE_INDEX" as a placeholder for features."""
        return torch.cat(self._tokens, dim=0)

    def concat_loss_mask(self) -> torch.BoolTensor:
        """Returns a vector of loss_mask."""
        return torch.cat(self._loss_mask, dim=0)

    def concat_features(self, embedding: nn.Embedding | None) -> torch.FloatTensor:
        """Returns concatenated features. Missing features are looked up with the given embedding module"""
        features = []
        for tokens, f in zip(self._tokens, self._features):
            if f is None:
                assert embedding is not None
                f = embedding(tokens)
            features.append(f)
        return torch.cat(features, dim=0)

    @staticmethod
    def batch(sequences: list[SequenceBuilder]):
        """
        Stack a list of equal-length sequences into a batch of tensors.
        Assume all tokens in the sequence come with features.

        Returns:
            tokens: shape is [N, seqlen]
            features: shape is [N, seqlen, embed_dim]
            loss_masks: shape is [N, seqlen]
        """
        lengths = [len(seq) for seq in sequences]
        assert len(set(lengths)) == 1, lengths

        tokens = torch.stack(
            [s.concat_tokens() for s in sequences], dim=0
        )  # [N, seqlen]
        features = torch.stack(
            [s.concat_features(None) for s in sequences], dim=0
        )  # [N, seqlen, dim]
        loss_mask = torch.stack(
            [s.concat_loss_mask() for s in sequences], dim=0
        )  # [N, seqlen]
        return tokens, features, loss_mask

    @staticmethod
    def cat(sequences: list[SequenceBuilder], keep_name=True) -> SequenceBuilder:
        """
        Concatenate a list of sequences into a single sequence.

        Args:
            keep_name: whether to keep name_to_range mapping of each sequence.
                If True, the given sequences cannot have duplicate names.
        """
        seq = SequenceBuilder()
        curr_len = 0
        for s in sequences:
            for token, feature, loss_mask in zip(s._tokens, s._features, s._loss_mask):
                seq.add_token_and_feature(
                    token=token, feature=feature, loss_mask=loss_mask
                )
            if keep_name:
                for name, (start, end) in s._name_to_range.items():
                    if name in seq._name_to_range:
                        raise KeyError(f"Name {name} already exists in the sequence.")
                    seq._name_to_range[name] = (start + curr_len, end + curr_len)
            curr_len += len(s)
        return seq

    @staticmethod
    def pad_to_longest(
        sequences: list[SequenceBuilder], padding_token: int, padding_side: str
    ) -> list[SequenceBuilder]:
        """
        Pad sequences to the longest among them all.
        Paddings will have zero features.

        Args:
            padding_token: the token id to be used as padding.
            padding_side: "left" or "right", where paddings are added.
        """
        assert padding_side in ["left", "right"]
        max_seqlen = max(len(seq) for seq in sequences)
        ret = []

        def cat(seq, padding):
            x = [seq, padding]
            return SequenceBuilder.cat(x if padding_side == "right" else x[::-1])

        for seq in sequences:
            padded_len = max_seqlen - len(seq)
            if padded_len > 0:
                pad_seq = SequenceBuilder()
                first_feature = seq._features[0]
                assert (
                    first_feature is not None
                ), "Feature is required in order to pad sequences."
                pad_seq.add_token_and_feature(
                    token=padding_token
                    * torch.ones(padded_len, device=seq.device, dtype=torch.long),
                    feature=first_feature.new_zeros(padded_len, first_feature.shape[1]),
                    loss_mask=False,
                )
                ret.append(cat(seq, pad_seq))
            else:
                ret.append(seq)
        return ret
