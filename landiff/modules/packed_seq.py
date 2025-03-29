import math
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F


class PackedSeqlens:
    """Maintain a list of seqlens and some cached properties.

    This class holds both CPU shapes and CUDA shapes. Each property tries
    to avoid CPU-CUDA copy as much as possible. If copy has to happen, result will
    be cached.
    """

    num_paddings: int

    # NOTE: https://discuss.python.org/t/finding-a-path-forward-for-functools-cached-property/23757
    # cached_property may have performance issues.

    # TODO: rename seqlens to shapes.
    def __init__(
        self,
        seqlens: list[int | tuple[int, ...] | torch.Size] | None = None,
        shapes_tensor: torch.Tensor | None = None,
        *,
        num_paddings: int = 0,
    ):
        """
        Args:
            seqlens: a list of int where each element is the length of one sequence.
                Or a list of shapes, where each is the (unflattened) shape, e.g. (h, w)
                of one flattened sequence.
            shapes_tensor: A NxD tensor containing the shapes of the sequences.
            num_paddings: the last `num_paddings` sequences are padding sequences.
        """
        assert (seqlens is not None) or (
            shapes_tensor is not None
        ), "One of seqlens and shaeps_tensor must be given."
        self.num_paddings = num_paddings
        self._cpu_shapes = None
        self._cuda_shapes = None

        if seqlens is not None:
            assert len(seqlens) > 0, seqlens
            shapes = []
            for s in seqlens:
                if isinstance(s, int):
                    shapes.append((s,))
                elif isinstance(s, (tuple, torch.Size)):
                    s = tuple(s)
                    assert all(isinstance(x, int) for x in s), s
                    shapes.append(s)
                else:
                    raise ValueError(
                        f"Invalid seqlen: {s}. Must be an int or a tuple of ints."
                    )

            dims = [len(x) for x in shapes]
            assert (
                len(set(dims)) == 1
            ), f"All shapes must have the same number of dimensions: {shapes}"
            self._cpu_shapes = torch.tensor(shapes, dtype=torch.int32, device="cpu")

        if shapes_tensor is not None:
            assert shapes_tensor.ndim == 2, shapes_tensor.shape
            assert shapes_tensor.dtype == torch.int32, shapes_tensor.dtype

            if seqlens is not None:
                # If both are provided, check consistency.
                self._check_input_consistency(shapes_tensor, self._cpu_shapes)

            if shapes_tensor.device.type == "cpu":
                self._cpu_shapes = shapes_tensor
            else:
                self._cuda_shapes = shapes_tensor
        if num_paddings > len(self):
            raise ValueError(
                f"Number of paddings {num_paddings} is more than the number of sequences {len(self)}!"
            )

    def _check_input_consistency(self, shapes1: torch.Tensor, shapes2: torch.Tensor):
        assert shapes1.shape == shapes2.shape, f"{shapes1.shape}, {shapes2.shape}"
        if shapes1.device.type == shapes2.device.type:
            if shapes1.device.type == "cpu":
                # CPU data, just check directly.
                assert (shapes1 == shapes2).all(), f"{shapes1}, {shapes2}"
            else:
                # CUDA data, launch an async check.
                torch._assert_async((shapes1 == shapes2).all())
        # Heterogeneous data, copy to CPU and check. Only do this if running unittests.

    def __len__(self) -> int:
        """Number of sequences in the pack."""
        if self._cpu_shapes is not None:
            return len(self._cpu_shapes)
        else:
            return len(self._cuda_shapes)

    # Basic methods; need data on CPU:
    def __str__(self):
        data_device_info = f" has_cpu_shape: {self._cpu_shapes is not None}, has_cuda_shape: {self._cuda_shapes is not None}"
        if self.shapes_tensor("cpu").shape[1] == 1:
            return f"PackedSeqlens({self.lengths()})." + data_device_info
        else:
            return f"PackedSeqlens({self.shapes()})." + data_device_info

    __repr__ = __str__

    def __hash__(self):
        return hash((tuple(self.shapes()), self.num_paddings))

    def shapes(self) -> list[tuple[int, ...]]:
        """Returns the shapes of the sequences."""
        return [tuple(x) for x in self.shapes_tensor("cpu").tolist()]

    def lengths(self) -> list[int]:
        """Returns the lengths of the sequences."""
        # Can add cache if needed.
        return [math.prod(s) for s in self.shapes()]

    @cached_property
    def _all_equal(self) -> bool:
        """Returns True if all sequences have the same length."""
        return len(set(self.lengths())) == 1

    def total_seqlen(self) -> int:
        """Returns the total length of all sequences."""
        return sum(self.lengths())

    def max_seqlen(self) -> int:
        """Returns the maximum lengths of all sequences."""
        return max(self.lengths())

    def min_seqlen(self) -> int:
        """Returns the minimum lengths of all sequences."""
        return min(self.lengths())

    def average_seqlen(self) -> int:
        """Returns the average lengths of all sequences."""
        return np.mean(self.lengths())

    def shapes_tensor(self, device):
        """Returns a cached tensor of shapes."""
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cpu":
            if self._cpu_shapes is None:
                self._cpu_shapes = self._cuda_shapes.cpu()
            return self._cpu_shapes
        if device.type == "cuda":
            if self._cuda_shapes is None:
                self._cuda_shapes = self._cpu_shapes.cuda()
            return self._cuda_shapes

    def tensor(self, device) -> torch.Tensor:
        """Returns a (N,) tensor of seqlens."""
        shapes_tensor = self.shapes_tensor(device)
        return shapes_tensor.prod(dim=1, dtype=torch.int32)

    def cu_seqlens(self, device) -> torch.Tensor:
        """Returns a cached (#seq+1,) vector of cumulative seqlens, to be used with FlashAttention v2."""
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            return self._cu_seqlens_cuda
        else:
            return self._cu_seqlens_cpu

    @cached_property
    def _cu_seqlens_cuda(self) -> torch.Tensor:
        if self._cuda_shapes is not None:
            # Use cuda data directly, if available.
            lengths_tensor = self.tensor("cuda")
            seqlens = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device="cuda"), lengths_tensor],
                dim=0,
            )
            return seqlens.cumsum(dim=0, dtype=torch.int32)

        if self._all_equal:
            # For equal-lengths sequences, directly construct results on CUDA.
            step = self.lengths()[0]
            return torch.arange(
                0, (len(self) + 1) * step, step, dtype=torch.int32, device="cuda"
            )

        # As a last resort, do a copy.
        return self._cu_seqlens_cpu.cuda()

    @cached_property
    def _cu_seqlens_cpu(self) -> torch.Tensor:
        seqlens = torch.tensor([0] + self.lengths(), dtype=torch.int32, device="cpu")
        return seqlens.cumsum(dim=0, dtype=torch.int32)

    # SeqLenInfo is not optimized. They are not used anymore.
    def SeqLenInfo(self, device):
        """Returns a xformers SeqLenInfo object from this seqlen."""
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            return self._SeqLenInfo_cuda
        else:
            return self._SeqLenInfo_cpu

    @cached_property
    def _SeqLenInfo_cpu(self):
        """Returns a xformers SeqLenInfo object from this seqlen."""
        from xformers.ops.fmha.attn_bias import _SeqLenInfo

        return _SeqLenInfo.from_seqlens(self.lengths())

    @cached_property
    def _SeqLenInfo_cuda(self):
        """Returns a xformers SeqLenInfo object from this seqlen."""
        from xformers.ops.fmha.attn_bias import _SeqLenInfo

        r = _SeqLenInfo.from_seqlens(self.lengths())
        r.to(device="cuda")
        return r

    def transform_shapes(self, transform_fn) -> "PackedSeqlens":
        """Transforms shapes using a provided function and returns a new PackedSeqlens instance.

        Args:
            transform_fn: Function to transform the shapes tensor.

        Returns:
            PackedSeqlens: A new instance of PackedSeqlens with transformed shapes or shapes_tensor.
        """
        new_seqlens = None
        new_cuda_tensor = None
        if self._cpu_shapes is not None:
            new_seqlens = [tuple(x) for x in transform_fn(self._cpu_shapes).tolist()]

        if self._cuda_shapes is not None:
            new_cuda_tensor = transform_fn(self._cuda_shapes)

        return PackedSeqlens(
            seqlens=new_seqlens,
            shapes_tensor=new_cuda_tensor,
            num_paddings=self.num_paddings,
        )

    def with_padding(self, *, shape: tuple[int, ...] | int) -> "PackedSeqlens":
        """Create a new PackedSeqlens with the given shape as padding."""
        shapes = self.shapes()
        if isinstance(shape, int):
            shape = tuple([shape] + [1] * (len(shapes[0]) - 1))
        shapes += [shape]
        num_paddings = self.num_paddings + 1
        return PackedSeqlens(shapes, num_paddings=num_paddings)

    def without_padding(self) -> "PackedSeqlens":
        """Create a new PackedSeqlens without padding."""
        new_cuda_shapes = None

        if self._cuda_shapes is not None:
            new_cuda_shapes = self._cuda_shapes[: -self.num_paddings]

        return PackedSeqlens(
            self.shapes()[: -self.num_paddings], shapes_tensor=new_cuda_shapes
        )

    def total_paddings(self) -> int:
        """Returns the total lengths of padding elements."""
        return sum(self.lengths()[-self.num_paddings :])

    def generate_dim_index_map(self, dim: int, device: torch.device | str):
        """Generates a tensor mapping 1D indices to the specified dimension's indices.

        Args:
            dim: Target dimension for mapping (e.g., 0: T, 1: H, 2: W for 3D input).
            device: Device for tensor allocation.

        Returns:
            Tensor of dimension indices corresponding to 1D indices.
        """
        if isinstance(device, str):
            device = torch.device(device)
        r = []
        for shape in self.shapes():
            r.append(
                torch.unravel_index(
                    torch.arange(int(np.prod(shape)), device=device), shape
                )[dim]
            )
        return torch.cat(r).to(dtype=torch.int32)

    def hw_per_doc(self, device: torch.device | str) -> torch.Tensor:
        """Calculate the product of height and width for each document.

        Args:
            device: The device to place the resulting tensor on.
        """
        if isinstance(device, str):
            device = torch.device(device)
        hw_per_doc = []
        for shape in self.shapes():
            hw_per_doc.extend([shape[-2] * shape[-1]])
        hw_per_doc = torch.tensor(hw_per_doc, device=device, dtype=torch.int32)
        return hw_per_doc

    ##############################
    # Methods that manipulate packed tensors with seqlens info.
    # Need seqlens data on CPU.
    def split(self, tensor: torch.Tensor, dim=-1) -> list[torch.Tensor]:
        """Splits a tensor by sequence lengths.

        Args:
            tensor: The tensor to split.
            dim: The dimension along which to split (default: 1).
        Returns:
            A list of tensors, where each element corresponds to a sequence
            in the original tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input tensor must be a torch.Tensor")
        splits = torch.split(tensor, self.lengths(), dim=dim)
        return splits

    # Need seqlens data on CPU and CUDA.
    # TODO: whether CPU data can be avoided?
    def repeat_interleave(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Returns x.repeat_interleave(self.tensor, dim=dim)"""
        device = x.device
        return x.repeat_interleave(self.tensor(device), dim=dim)

    def pad(
        self, x: torch.Tensor | None, dim: int = 0, value: float = 0
    ) -> torch.Tensor:
        """Adds paddings to a tensor.

        Args:
            x: The tensor to pad. Its shape along dim must match the valid parts of self.
            dim: The dimension along which to pad (default: 0).
            value: The padding value (default: 0).
        Returns:
            A new tensor with padding applied.
        """
        if self.num_paddings == 0 or x is None:
            return x
        size = x.shape[dim]
        assert size == sum(
            self.lengths()[: -self.num_paddings]
        ), f"{x.shape}, {self.shapes()}"
        paddings = self.total_paddings()
        pad_spec = [0, 0] * ((x.ndim - dim - 1) % x.ndim) + [0, paddings]
        return F.pad(x, pad_spec, mode="constant", value=value)
