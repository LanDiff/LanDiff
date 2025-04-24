from functools import lru_cache

import torch


class PackedSeqlens:
    def __init__(self, seqlens: list[int]):
        self.seqlens = seqlens

    @lru_cache()
    def cu_seqlens(self, device) -> torch.Tensor:
        """Returns a cached (#seq+1,) vector of cumulative seqlens, to be used with FlashAttention v2."""
        if isinstance(device, str):
            device = torch.device(device)
        seqlens = torch.tensor([0] + self.seqlens, dtype=torch.int32, device=device)
        return seqlens.cumsum(dim=0, dtype=torch.int32)

    def total_seqlen(self) -> int:
        """Returns the total length of all sequences."""
        return sum(self.seqlens)

    def max_seqlen(self) -> int:
        """Returns the maximum lengths of all sequences."""
        return max(self.seqlens)
