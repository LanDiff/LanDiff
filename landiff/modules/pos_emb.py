import os
from functools import cached_property

import einops
import torch
from torch import nn


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    rope_apply_type = os.environ.get("ROPE_APPLY_TYPE", "torch")

    if rope_apply_type == "torch":
        freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
        xq_ = torch.view_as_complex(
            xq.float().view(*xq.shape[:-1], -1, 2)
        )  # ..., num_heads, head_dim/2
        xk_ = torch.view_as_complex(
            xk.float().view(*xq.shape[:-1], -1, 2)
        )  # ..., num_heads, head_dim/2
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(
            -2
        )  # ..., num_heads, head_dim
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(
            -2
        )  # ..., num_heads, head_dim
        return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    interpolation_factor: int = 1,
    max_seq_length: int = 4096,
):
    print(
        f"using rope base theta = {theta}, interpolation factor = {interpolation_factor}"
    )
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device).float()
    scale = 1.0 / float(interpolation_factor)
    t *= scale

    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    if max_seq_length < end:
        freqs_cis = freqs_cis[:max_seq_length,].clone()
    return freqs_cis


class Rope1DPosEmb(nn.Module):

    def __init__(
        self,
        dim: int,
        max_len: int,
        theta_base=10000,
        device="cuda",
    ):
        """1D rotary position embedding.

        Args:
            dim: usually the multi-head attention dimension, should be divisible by 2
            max_len: the maximum sequence length.
        """
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "dim must be divisible by 2"
        self.max_len = max_len
        self.theta_base = theta_base
        self.device = device

    def extra_repr(self):
        return f"dim={self.dim}, max_len={self.max_len}, theta_base={self.theta_base}"

    @cached_property
    def precomputed_freqs_cis(self) -> torch.Tensor:
        cis = precompute_freqs_cis(
            dim=self.dim,
            end=self.max_len,
            theta=self.theta_base,
            max_seq_length=self.max_len,
        )
        return cis.to(self.device)

    def get_freqs_cis_by_seqlens(
        self,
        seqlens: list[int],
    ) -> torch.Tensor:
        """
        Args:
            seqlens: containing list of sequence lengths.
        Return:
            freqs_cis: tensor of shape (sum(seqlens), dim//2)
        """
        assert all(1 <= s <= self.max_len for s in seqlens), (
            seqlens,
            self.max_len,
        )
        freqs_cis = torch.cat([self.precomputed_freqs_cis[:s] for s in seqlens], dim=0)
        return freqs_cis


class Rope3DPosEmb(nn.Module):

    def __init__(
        self,
        dim: int,
        max_time: int,
        max_height: int,
        max_width: int,
        one_dim_max_time: int | None = None,
        multiple: int = 6,
        theta_base=10000,
        device="cuda",
    ):
        """Args:
        dim: usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height, max_width: the maximum height and width of the 2D grid
        """
        super().__init__()
        self.dim = dim
        self.multiple = multiple
        assert self.multiple == 6 or self.multiple == 16, "multiple must be 6 or 16"
        assert (
            self.dim % self.multiple == 0
        ), f"dim must be divisible by {self.multiple}, but got {self.dim}"
        self.max_time = max_time
        self.max_height = max_height
        self.max_width = max_width
        if one_dim_max_time is None:
            self.one_dim_max_time = max_time
        else:
            self.one_dim_max_time = one_dim_max_time
        self.theta_base = theta_base
        self.device = device

    def extra_repr(self):
        return f"dim={self.dim}, max_time={self.max_time}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def get_text_freqs_cis(self):
        t_pos = torch.arange(0, self.one_dim_max_time).float().to(self.device)
        h_pos = torch.arange(0, self.one_dim_max_time).float().to(self.device)
        w_pos = torch.arange(0, self.one_dim_max_time).float().to(self.device)
        if self.multiple == 6:
            dim_range = (
                torch.arange(0, self.dim, 6)[: (self.dim // 6)].float().to(self.device)
            )  # C/6
            freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
            t_freqs = torch.outer(t_pos, freqs).float()  # N, C/6
            h_freqs = torch.outer(h_pos, freqs).float()  # N, C/6
            w_freqs = torch.outer(w_pos, freqs).float()  # N, C/6
        else:
            t_dim = self.dim // 4
            h_w_dim = self.dim // 8 * 3
            t_dim_range = (
                torch.arange(0, t_dim, 2)[: (t_dim // 2)].float().to(self.device)
            )  # C/8
            h_w_dim_range = (
                torch.arange(0, h_w_dim, 2)[: (h_w_dim // 2)].float().to(self.device)
            )  # C/16*3
            t_freqs = 1.0 / (self.theta_base ** (t_dim_range / t_dim))
            t_freqs = torch.outer(t_pos, t_freqs).float()  # N, C/8
            h_w_freqs = 1.0 / (self.theta_base ** (h_w_dim_range / h_w_dim))
            h_freqs = torch.outer(h_pos, h_w_freqs).float()  # N, C/16*3
            w_freqs = torch.outer(w_pos, h_w_freqs).float()  # N, C/16*3
        t_cis = torch.polar(torch.ones_like(t_freqs), t_freqs)  # N, C/6
        h_cis = torch.polar(torch.ones_like(h_freqs), h_freqs)  # N, C/6
        w_cis = torch.polar(torch.ones_like(w_freqs), w_freqs)  # N, C/6
        if self.multiple == 6:
            freqs_cis = torch.cat(
                [
                    t_cis.unsqueeze(dim=-1),
                    h_cis.unsqueeze(dim=-1),
                    w_cis.unsqueeze(dim=-1),
                ],
                dim=-1,
            )  # N, C/6, 3
        else:
            freqs_cis = torch.cat([t_cis, h_cis, w_cis], dim=-1)  # N, C/2
        freqs_cis = freqs_cis.reshape(self.one_dim_max_time, -1)
        return freqs_cis

    @cached_property
    def precomputed_freqs_cis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            time axis: ret[t, h, w, 2*i] = cis(t * theta_base**(-6*i/dim))
            height axis: ret[t, h, w, 2*i+1] = cis(h * theta_base**(-6*i/dim))
            weight axis: ret[t, h, w, 2*i+2] = cis(w * theta_base**(-6*i/dim))   with (i in [0, dim//6))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_time * self.max_height * self.max_width
        flat_pos = (
            torch.arange(0, N).float().to(self.device)
        )  # flat_pos = t_pos * (max_height * max_width) + h_pos * max_width + w_pos
        t_pos = flat_pos // (self.max_height * self.max_width)
        h_pos = (flat_pos % (self.max_height * self.max_width)) // self.max_width
        w_pos = (flat_pos % (self.max_height * self.max_width)) % self.max_width
        if self.multiple == 6:
            dim_range = (
                torch.arange(0, self.dim, 6)[: (self.dim // 6)].float().to(self.device)
            )  # C/6
            freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
            t_freqs = torch.outer(t_pos, freqs).float()  # N, C/6
            h_freqs = torch.outer(h_pos, freqs).float()  # N, C/6
            w_freqs = torch.outer(w_pos, freqs).float()  # N, C/6
        else:
            t_dim = self.dim // 4
            h_w_dim = self.dim // 8 * 3
            t_dim_range = (
                torch.arange(0, t_dim, 2)[: (t_dim // 2)].float().to(self.device)
            )  # C/8
            h_w_dim_range = (
                torch.arange(0, h_w_dim, 2)[: (h_w_dim // 2)].float().to(self.device)
            )  # C/16*3
            t_freqs = 1.0 / (self.theta_base ** (t_dim_range / t_dim))
            t_freqs = torch.outer(t_pos, t_freqs).float()  # N, C/8
            h_w_freqs = 1.0 / (self.theta_base ** (h_w_dim_range / h_w_dim))
            h_freqs = torch.outer(h_pos, h_w_freqs).float()  # N, C/16*3
            w_freqs = torch.outer(w_pos, h_w_freqs).float()  # N, C/16*3
        t_cis = torch.polar(torch.ones_like(t_freqs), t_freqs)  # N, C/6
        h_cis = torch.polar(torch.ones_like(h_freqs), h_freqs)  # N, C/6
        w_cis = torch.polar(torch.ones_like(w_freqs), w_freqs)  # N, C/6
        if self.multiple == 6:
            freqs_cis = torch.cat(
                [
                    t_cis.unsqueeze(dim=-1),
                    h_cis.unsqueeze(dim=-1),
                    w_cis.unsqueeze(dim=-1),
                ],
                dim=-1,
            )  # N, C/6, 3
        else:
            freqs_cis = torch.cat([t_cis, h_cis, w_cis], dim=-1)
        freqs_cis = freqs_cis.reshape(
            self.max_time, self.max_height, self.max_width, -1
        )  # max_time, max_height, max_width, C/2
        text_freqs_cis = self.get_text_freqs_cis()
        return freqs_cis, text_freqs_cis

    def get_freqs_cis_by_idx(
        self, pos_idx: torch.Tensor, pos_idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_idx: tensor of shape (..., 3), It contains the (t, h, w) position indices of each 3D token.
            pos_idx_mask: a mask of shape (...), the leading dimensions should be the same as pos_idx.
                Rope will only be applied to the tokens with True mask. `freqs_cis` for the tokens with False mask with be ones.
        Return:
            freqs_cis: tensor of shape (..., dim//2)
        """
        assert (
            pos_idx.shape[:-1] == pos_idx_mask.shape
            and pos_idx.shape[-1] == 3
            and pos_idx.ndim == pos_idx_mask.ndim + 1
        ), (pos_idx.shape, pos_idx_mask.shape)
        assert pos_idx_mask.dtype == torch.bool, pos_idx_mask.dtype
        pos_idx = pos_idx.to(self.device)
        pos_idx_mask = pos_idx_mask.to(self.device)
        shp = pos_idx_mask.shape + (self.dim // 2,)  # ..., head_dim/2
        freqs_cis = torch.ones(
            shp, dtype=torch.complex64, device=self.device
        )  # ..., head_dim/2
        pre_freqs_cis, pre_text_freqs_cis = self.precomputed_freqs_cis

        used_pos_id = pos_idx[pos_idx_mask].reshape(-1, 3)
        used_freqs_cis = torch.ones(
            [used_pos_id.shape[0], self.dim // 2],
            dtype=torch.complex64,
            device=self.device,
        )
        # 判断哪些位置的pos_id, 最后一个维度的三个数字都是相等的
        equal_pos_id_mask = (used_pos_id[:, 0] == used_pos_id[:, 1]) & (
            used_pos_id[:, 1] == used_pos_id[:, 2]
        )

        equal_pos_id = used_pos_id[equal_pos_id_mask].reshape(-1, 3)
        not_equal_pos_id = used_pos_id[~equal_pos_id_mask].reshape(-1, 3)
        not_equal_pos_freqs_cis = pre_freqs_cis[
            not_equal_pos_id[..., 0], not_equal_pos_id[..., 1], not_equal_pos_id[..., 2]
        ]
        equal_pos_freqs_cis = pre_text_freqs_cis[equal_pos_id[..., 0]]
        used_freqs_cis[equal_pos_id_mask] = equal_pos_freqs_cis
        used_freqs_cis[~equal_pos_id_mask] = not_equal_pos_freqs_cis

        freqs_cis[pos_idx_mask] = used_freqs_cis
        return freqs_cis

    @staticmethod
    def shape_to_index(
        t: int, h: int, w: int, device: torch.device
    ) -> torch.LongTensor:
        """将给定的三维形状 (t, h, w) 转换为包含所有可能三维索引的张量。

        参数:
        t (int): 时间维度的大小。
        h (int): 高度维度的大小。
        w (int): 宽度维度的大小。
        device (torch.device): 生成张量的设备。

        返回:
        torch.LongTensor: 一个形状为 (t*h*w, 3) 的长整型张量，其中每一行表示一个三维索引。

        示例:
        >>> t, h, w = 2, 2, 2
        >>> device = torch.device('cpu')
        >>> shape_to_index(t, h, w, device)
        tensor([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]])
        """
        t_idx = torch.arange(t, device=device, dtype=torch.long)
        h_idx = torch.arange(h, device=device, dtype=torch.long)
        w_idx = torch.arange(w, device=device, dtype=torch.long)
        tt, hh, ww = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        rope_idx = torch.stack([tt, hh, ww], dim=-1).reshape(-1, 3)

        return rope_idx

    @staticmethod
    def shift_rope_index(
        rope_index: torch.LongTensor, shift: int, shift_all=False
    ) -> tuple[torch.LongTensor, int]:
        """对给定的 rope_index 张量进行位移操作，并返回新的张量和新的位移值。

        参数:
        rope_index (torch.LongTensor): 输入的张量，形状为 (N, M)。
        shift (int): 位移的值。
        shift_all (bool, 可选): 如果为 True，则对整个张量的所有元素进行位移操作。
                                如果为 False，则仅对张量的第一列进行位移操作。默认为 False。

        返回:
        tuple[torch.LongTensor, int]: 包含两个元素的元组，
                                      第一个元素是位移后的新张量，
                                      第二个元素是新位移值，为新张量最后一行第一列元素加 1 的结果。

        示例:
        >>> rope_index = torch.LongTensor([[0, 0, 0],
                                           [0, 0, 1],
                                           [0, 1, 0],
                                           [0, 1, 1],
                                           [1, 0, 0],
                                           [1, 0, 1],
                                           [1, 1, 0],
                                           [1, 1, 1]])
        >>> shift = 2
        >>> shift_all = False
        >>> new_rope_index, new_shift = YourClassName._shift_rope_index(rope_index, shift, shift_all)
        >>> print(new_rope_index)
        tensor([[2, 0, 0],
                [2, 0, 1],
                [2, 1, 0],
                [2, 1, 1],
                [3, 0, 0],
                [3, 0, 1],
                [3, 1, 0],
                [3, 1, 1]])
        >>> print(new_shift)
        4
        """
        new_rope_index = rope_index.clone()
        if shift_all:
            new_rope_index = new_rope_index + shift
        else:
            new_rope_index[:, 0] += shift
        new_shift = int(new_rope_index[-1, 0] + 1)
        return new_rope_index, new_shift

    @staticmethod
    def len_to_rope_index(n, device) -> torch.LongTensor:
        """生成一个二维张量，其中每一行都是从 0 到 n-1 的序列，重复 c 次。

        参数:
        n (int): 生成张量的长度。
        device (str): 指定张量所在的设备（例如 'cpu' 或 'cuda'）。

        返回:
        torch.LongTensor: 一个形状为 (n, c) 的二维张量，其中每一行都是从 0 到 n-1 的序列，重复 c 次。

        示例:
        >>> PosEmb.len_to_rope_index(5, 'cpu')
        tensor([[0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4]])
        """
        rope_index = torch.arange(0, n, device=device, dtype=torch.long)
        rope_index = einops.repeat(rope_index, "n -> n c", c=3)
        return rope_index
