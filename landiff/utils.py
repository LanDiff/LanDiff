import contextlib
import hashlib
from contextlib import contextmanager
from pathlib import Path

import einops
import imageio
import numpy as np
import torch
from torch import nn


class _FreezeSentinel:
    pass


def freeze_model(
    model: nn.Module, disable_state_dict=True, disable_grad=True, disable_train=True
):
    """Freeze a model.

    Args:
        disable_state_dict: make it stateless to save checkpoint space.
            If True, model's state_dict will have None values and load_state_dict will not issue any
            warnings or errors. Normally you should have already loaded a state dict to
            it before calling this function.
        disable_grad: disable gradient for all its parameters
        disable_train: disable train mode

    Note that this method does not disable autograd when calling model.
    """

    if disable_state_dict:
        # Hook load_state_dict to prevent it from adding `missing_keys`. This way, the model
        # can load an empty state-dict without errors. To achieve this, we insert a sentinel
        # in prehook, and clear all items after the sentinel in posthook.
        def prehook(_, __, ___, ____, missing_keys, unexpected_keys, _____):
            missing_keys.append(_FreezeSentinel)

        def posthook(_, incompatible):
            missing = incompatible.missing_keys
            while missing[-1] is not _FreezeSentinel:
                missing.pop()
            missing.pop()

        model._register_load_state_dict_pre_hook(prehook)
        model.register_load_state_dict_post_hook(posthook)

        orig_state_dict_func = model.state_dict

        def state_dict(*args, **kwargs):
            if "destination" in kwargs:
                old_keys = set(kwargs["destination"].keys())
            else:
                old_keys = set()
            sd = orig_state_dict_func(*args, **kwargs)
            # For every new key that `orig_state_dict_func` added,
            # set the value to None so it does not take space.
            # We cannot pop it because FSDP will raise error in it's post_state_dict_hook
            # if model.state_dict() does not match the model's named_parameters.
            for k in set(sd.keys()) - old_keys:
                sd[k] = None
            return sd

        model.state_dict = state_dict

    if disable_grad:
        model.requires_grad_(False)

    if disable_train:
        model.eval()
        model.train = lambda mode=True: None


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


@contextmanager
def maybe_autocast(tensor, dtype=None, cache_enabled=None):
    """If dtype is None, open an autocast context with tensor.dtype if it's not float.

    If dtype is not None, open an autocast context with tensor.dtype if it matches dtype.

    cache_enabled should be True when training, so no cast will happen in backward.
    Default to is_grad_enabled().
    """
    if dtype is not None:
        assert dtype in [torch.bfloat16, torch.float16]
        enabled = tensor.dtype == dtype
    else:
        enabled = tensor.dtype in [torch.bfloat16, torch.float16]
        dtype = tensor.dtype if enabled else torch.bfloat16
    if cache_enabled is None:
        cache_enabled = torch.is_grad_enabled()
    with contextlib.ExitStack() as cm:
        # Enter context on both devices to avoid confusion.
        # Otherwise models may raise a "dtype mismatch" error when
        # the actual reason is device mismatch, causing autocast to be inactive.
        if enabled:
            if dtype == torch.bfloat16:
                cm.enter_context(
                    torch.autocast("cpu", dtype=dtype, cache_enabled=cache_enabled)
                )
            if torch.cuda.is_available():
                cm.enter_context(
                    torch.autocast("cuda", dtype=dtype, cache_enabled=cache_enabled)
                )
        yield


def stable_hash(key: str) -> int:
    """Generate a stable hash from a key.

    Python's built-in hash() function is not stable across runs.
    """
    hex_hash = hashlib.sha256(key.encode()).hexdigest()
    hex_hash = hex_hash[:20]  # take the first 20 digits to reduce computational cost
    return int(hex_hash, 16)


def cthw_to_numpy_images(video: torch.Tensor) -> np.ndarray:
    assert video.dim() == 4, "video must be 4D tensor"
    images = einops.rearrange(video, "c t h w -> t h w c") * 255
    images = images.clip(0, 255).cpu().numpy().astype(np.uint8)
    return images


def save_video_tensor(video: torch.Tensor, video_path: str, fps: int = 8):
    assert video.dim() == 4, "video must be 4D tensor"
    video_images = cthw_to_numpy_images(video)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with open(video_path, "wb") as f:
        with imageio.get_writer(f, format="mp4", fps=fps) as writer:
            for image in video_images:
                writer.append_data(image)


def top_p_probability(top_p: float, probs: torch.Tensor):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_idx_remove_cond = cum_probs >= top_p

    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0

    indices_to_remove = sorted_idx_remove_cond.scatter(
        -1, sorted_indices, sorted_idx_remove_cond
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return probs


def top_k_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, k: int = 1
) -> torch.Tensor:
    """Computes the top-k accuracy for the given outputs and labels.

    Args:
        outputs: The outputs in shape [..., C] where C is number classes.
        labels: labels of shape [...], integer in range [0, C-1].
    """
    _, top_k_predictions = outputs.topk(k, dim=-1)
    correct = top_k_predictions.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean()


def assign_module_scope(model: nn.Module, prefix=""):
    """Assign a `.__scope__` attribute to every submodule of model, using the
    module's dot-separated name, e.g. `encoder.blocks.0.conv0`. This will allow
    each module to know its own name, which is useful for logging.

    Args:
        prefix: a prefix to append to all scopes, e.g. "modelA."
    """
    to_remove = ["_fsdp_wrapped_module", "_checkpoint_wrapped_module"]
    for name, m in model.named_modules():
        parts = name.split(".")
        parts = [k for k in parts if k not in to_remove]
        name = ".".join(parts)
        m.__scope__ = prefix + name


def maybe_assign_module_scope(model: nn.Module, prefix=""):
    """If none of the submodules has the `.__scope__` attribute, assign it to
    every submodule of the model.

    If some of submodules have this attribute while the others do not, raise an error.
    """
    module_with_scope = [
        True if hasattr(module, "__scope__") else False for module in model.modules()
    ]
    if not all(module_with_scope) and any(module_with_scope):
        raise ValueError(
            "Some of the submodules in the model have `.__scope__` attribute while others do not."
        )
    if not any(module_with_scope):
        assign_module_scope(model, prefix)


def set_seed_for_single_process(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
