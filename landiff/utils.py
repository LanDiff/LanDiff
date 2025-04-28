import contextlib
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import einops
import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from tqdm import tqdm

# Global variable to store model path
_LANDIFF_MODEL_PATH = None


class _FreezeSentinel:
    pass


def verify_md5_checksum(root_dir: Path) -> bool:
    # Get the root path of working directory
    work_dir = Path(__file__).resolve().parents[1]

    # Checksum file path is fixed as ckpts/CHECKSUM.md5
    checksum_file = work_dir / "ckpts" / "CHECKSUM.md5"
    if not checksum_file.exists():
        raise FileNotFoundError(f"Checksum file does not exist: {checksum_file}")

    # Read checksums
    checksums = {}
    with open(checksum_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            md5, filepath = line.split("  ", 1)
            # Remove ./ prefix from path
            if filepath.startswith("./"):
                filepath = filepath[2:]
            checksums[filepath] = md5

    # Verify files
    all_files_valid = True
    for rel_path, expected_md5 in tqdm(
        checksums.items(), desc="Verifying files", unit="files"
    ):
        # Calculate actual file location in the model directory
        file_path = root_dir / rel_path

        if not file_path.exists():
            print(f"Error: File does not exist: {file_path}")
            all_files_valid = False
            break

        # Calculate MD5 with progress bar for large files
        file_md5 = hashlib.md5()
        file_size = file_path.stat().st_size
        # Only show progress bar for files larger than 10MB
        if file_size > 10 * 1024 * 1024:  # 10MB
            chunk_size = (
                4096 * 256
            )  # Increase chunk size for better performance with large files
            with open(file_path, "rb") as f:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Checking {rel_path}",
                    leave=False,
                ) as pbar:
                    for chunk in iter(lambda: f.read(chunk_size), b""):
                        file_md5.update(chunk)
                        pbar.update(len(chunk))
        else:
            # For small files, don't show progress bar
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_md5.update(chunk)

        if file_md5.hexdigest() != expected_md5:
            print(f"Error: File verification failed: {file_path}")
            print(f"  Expected MD5: {expected_md5}")
            print(f"  Actual MD5: {file_md5.hexdigest()}")
            all_files_valid = False
            break

    return all_files_valid


def initialize_landiff_model_path():
    """Initialize and ensure availability of LanDiff model files

    This function manages the complete setup process for the LanDiff model:
    1. Locates model path using priorities:
       - LANDIFF_HOME environment variable
       - The ckpts/LanDiff path in the working directory
    2. Downloads the model from Hugging Face if not found locally
    3. Validates model files using MD5 checksums
    4. Creates symbolic links for consistent path access

    When a valid model path is found, the function will create a symbolic link from
    the workspace path (ckpts/LanDiff) to the actual model path if they are different.
    This ensures that the model can be consistently accessed through the workspace path.

    After the first call, subsequent calls will return the cached path directly.

    Returns:
        Path: Initialized and validated model files storage path

    Raises:
        FileExistsError: When workspace_path exists and is not a symlink, to prevent
                         accidental deletion of user data
        FileNotFoundError: When checksum file does not exist
        ValueError: When hash verification of the downloaded model fails
    """
    global _LANDIFF_MODEL_PATH

    # If already validated, return directly
    if _LANDIFF_MODEL_PATH is not None:
        return _LANDIFF_MODEL_PATH

    # Get root path of working directory
    root_dir = Path(__file__).resolve().parents[1]

    # Check possible model paths
    potential_paths = []

    # 1. Check environment variable
    env_path = os.environ.get("LANDIFF_HOME")
    if env_path:
        potential_paths.append(Path(env_path))

    # 2. Check path in the working directory
    workspace_path = root_dir / "ckpts" / "LanDiff"
    potential_paths.append(workspace_path)

    # Check if the path exists and validate md5
    for model_path in potential_paths:
        if (
            model_path.exists()
            and model_path.is_dir()
            and verify_md5_checksum(model_path)
        ):
            _LANDIFF_MODEL_PATH = model_path

            # Create a symbolic link to workspace_path if the model is not already there
            if model_path != workspace_path:
                # Check if workspace_path exists and is not a symlink
                if workspace_path.exists() and not workspace_path.is_symlink():
                    raise FileExistsError(
                        f"Workspace path '{workspace_path}' already exists and is not a symbolic link. "
                        f"Please remove or rename it manually to create a symbolic link to the model path '{model_path}'."
                    )

                # Remove existing symbolic link if it exists
                if workspace_path.exists() and workspace_path.is_symlink():
                    workspace_path.unlink()

                # Create parent directory if it doesn't exist
                workspace_path.parent.mkdir(parents=True, exist_ok=True)

                # Create symbolic link
                workspace_path.symlink_to(model_path, target_is_directory=True)
                print(f"Created symbolic link from {workspace_path} to {model_path}")

            return model_path

    # If no valid model path is found, notify the user that the model will be automatically downloaded
    print(
        "No valid model path found. Will automatically download LanDiff model from Hugging Face..."
    )

    # Use snapshot_download to download the entire model repository
    download_path = Path(snapshot_download(repo_id="yinaoxiong/LanDiff"))

    print(f"Model downloaded to {download_path}, performing hash verification...")

    # Verify the downloaded model with hash checksum
    if verify_md5_checksum(download_path):
        print("Model hash verification successful!")
        _LANDIFF_MODEL_PATH = download_path

        # Create a symbolic link to workspace_path
        if download_path != workspace_path:
            # Check if workspace_path exists and is not a symlink
            if workspace_path.exists() and not workspace_path.is_symlink():
                raise FileExistsError(
                    f"Workspace path '{workspace_path}' already exists and is not a symbolic link. "
                    f"Please remove or rename it manually to create a symbolic link to the downloaded model path '{download_path}'."
                )

            # Remove existing symbolic link if it exists
            if workspace_path.exists() and workspace_path.is_symlink():
                workspace_path.unlink()

            # Create parent directory if it doesn't exist
            workspace_path.parent.mkdir(parents=True, exist_ok=True)

            # Create symbolic link
            workspace_path.symlink_to(download_path, target_is_directory=True)
            print(f"Created symbolic link from {workspace_path} to {download_path}")
        return download_path
    else:
        # If verification fails, raise an error
        raise ValueError(
            "Hash verification of the downloaded model failed. Please ensure a stable network connection, "
            "or manually download the model and set the LANDIFF_HOME environment variable."
        )


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
