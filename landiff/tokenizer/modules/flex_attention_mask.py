import time

import torch
import torch.nn.attention.flex_attention as flex_attention_mod
from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from tqdm import tqdm


def noop(score, b, h, q_idx, kv_idx):
    return score


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def generate_prefix_lm_mask(prefix_length: int) -> _mask_mod_signature:
    """Generates a prefix LM causal attention mask.

    Args:
        prefix_length: The length of the prefix.

    Note:
        This mask allows full attention within the prefix (first PREFIX_LENGTH tokens)
        and causal attention for the rest of the sequence.
    """

    def prefix_mask(b, h, q_idx, kv_idx):
        return kv_idx < prefix_length

    prefix_lm_causal_mask = or_masks(prefix_mask, causal)
    prefix_lm_causal_mask.__name__ = f"prefix_lm_causal_mask_{prefix_length}"
    return prefix_lm_causal_mask


class VideoEncoderMask:

    def __init__(
        self,
        num_frames: int,
        tokens_per_frame: int,
        IFrame_tokens: int,
        PFrame_tokens: int,
        block_size: int = 128,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.IFrame_tokens = IFrame_tokens
        self.PFrame_tokens = PFrame_tokens
        self.seq_len = (
            self.num_frames * self.tokens_per_frame
            + self.IFrame_tokens
            + self.PFrame_tokens * (self.num_frames - 1)
        )
        self.block_size = block_size
        self.device = device
        self.mask = self.create_mask()

    def slow_create_mask(self, q_len, kv_len):
        mask = torch.zeros((q_len, kv_len), dtype=torch.bool, device=self.device)
        bar = tqdm(total=q_len * kv_len)
        for q_idx in range(self.seq_len):
            for kv_idx in range(self.seq_len):
                mask[q_idx, kv_idx] = self._mask_fn(1, 1, q_idx, kv_idx)
                bar.update(1)
        bar.close()
        return mask

    def create_mask(self, use_vmap=True):
        # block mask 要求mask是block的倍数，所以这里要对mask进行padding否则在构造block mask的时候会index error
        if self.seq_len < 128:
            q_len = self.seq_len
        else:
            q_len = flex_attention_mod._round_up_to_multiple(
                self.seq_len, self.block_size
            )
        kv_len = flex_attention_mod._round_up_to_multiple(self.seq_len, self.block_size)
        if not use_vmap:
            mask = self.slow_create_mask(q_len, kv_len)
        else:
            mask = flex_attention_mod.create_mask(
                self.vmap_fn, B=1, H=1, Q_LEN=q_len, KV_LEN=kv_len, device=self.device
            )[0, 0, :, :]
        return mask

    def test_mask(self):
        no_vmap_start = time.time()
        no_vmap_mask = self.create_mask(use_vmap=False)
        no_vmap_end = time.time()
        vmap_start = time.time()
        vmap_mask = self.create_mask(use_vmap=True)
        vmap_end = time.time()
        print(f"no vmap time: {no_vmap_end - no_vmap_start}")
        print(f"vmap time: {vmap_end - vmap_start}")
        torch.testing.assert_close(no_vmap_mask, vmap_mask)

    def _to_bool_tensor(self, x):
        return torch.tensor(x, dtype=torch.bool, device=self.device)

    def _bool_warp_where(self, x, y, z):
        if not isinstance(x, torch.Tensor):
            x = self._to_bool_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = self._to_bool_tensor(y)
        if not isinstance(z, torch.Tensor):
            z = self._to_bool_tensor(z)
        return torch.where(x, y, z)

    def vmap_fn(self, b, h, q_idx, kv_idx):
        frame_idx = (
            q_idx - self.IFrame_tokens - self.num_frames * self.tokens_per_frame
        ) // self.PFrame_tokens + 1
        return self._bool_warp_where(
            q_idx < self.num_frames * self.tokens_per_frame,
            (q_idx // self.tokens_per_frame) >= (kv_idx // self.tokens_per_frame),
            self._bool_warp_where(
                q_idx < self.num_frames * self.tokens_per_frame + self.IFrame_tokens,
                self._bool_warp_where(
                    kv_idx < self.tokens_per_frame,
                    True,
                    self._bool_warp_where(
                        (kv_idx >= self.num_frames * self.tokens_per_frame)
                        & (
                            kv_idx
                            < self.num_frames * self.tokens_per_frame
                            + self.IFrame_tokens
                        ),
                        kv_idx <= q_idx,
                        False,
                    ),
                ),
                self._bool_warp_where(
                    q_idx < self.seq_len,
                    self._bool_warp_where(
                        kv_idx < (frame_idx + 1) * self.tokens_per_frame,
                        True,
                        self._bool_warp_where(
                            (kv_idx >= (frame_idx + 1) * self.tokens_per_frame)
                            & (kv_idx < self.tokens_per_frame * self.num_frames),
                            False,
                            kv_idx <= q_idx,
                        ),
                    ),
                    False,
                ),
            ),
        )

    def _mask_fn(self, b, h, q_idx, kv_idx):
        if q_idx < self.num_frames * self.tokens_per_frame:
            # If q_idx is within the range of the video frames
            frame_idx = q_idx // self.tokens_per_frame
            k_frame_idx = kv_idx // self.tokens_per_frame
            return k_frame_idx <= frame_idx  # Only allow attention to previous frames
        # I frame tokens can attend to first Frame and all previous tokens
        elif q_idx < self.num_frames * self.tokens_per_frame + self.IFrame_tokens:
            if kv_idx < self.tokens_per_frame:
                return True  # I frame tokens can attend to first Frame
            elif (
                kv_idx >= self.num_frames * self.tokens_per_frame
                and kv_idx
                < self.num_frames * self.tokens_per_frame + self.IFrame_tokens
            ):
                return (
                    kv_idx <= q_idx
                )  # I frame tokens can attend to all previous tokens
            else:
                return False  # I frame tokens can't attend to P frame tokens
        elif q_idx < self.seq_len:  # P frame tokens
            frame_idx = (
                q_idx - self.IFrame_tokens - self.num_frames * self.tokens_per_frame
            ) // self.PFrame_tokens + 1
            if kv_idx < (frame_idx + 1) * self.tokens_per_frame:
                return True  # P frame tokens can attend to previous Frame
            elif (
                kv_idx >= (frame_idx + 1) * self.tokens_per_frame
                and kv_idx < self.tokens_per_frame * self.num_frames
            ):
                return False  # P frame tokens can't attend to right side of the Frame
            else:
                return kv_idx <= q_idx
        else:
            return False

    def __call__(self, b, h, q_idx, kv_idx):
        q_idx_device = q_idx.device
        if self.mask.device != q_idx.device:
            self.mask = self.mask.to(q_idx_device)
        return self.mask[q_idx, kv_idx]


class VideoDecoderMask(VideoEncoderMask):

    def _to_bool_tensor(self, x):
        return torch.tensor(x, dtype=torch.bool, device=self.device)

    def _bool_warp_where(self, x, y, z):
        if not isinstance(x, torch.Tensor):
            x = self._to_bool_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = self._to_bool_tensor(y)
        if not isinstance(z, torch.Tensor):
            z = self._to_bool_tensor(z)
        return torch.where(x, y, z)

    def vmap_fn(self, b, h, q_idx, kv_idx):
        frame_idx = q_idx // self.tokens_per_frame
        kv_frame_idx = kv_idx // self.tokens_per_frame
        token_frame_idx = (
            q_idx - self.IFrame_tokens - self.num_frames * self.tokens_per_frame
        ) // self.PFrame_tokens + 1
        return self._bool_warp_where(
            q_idx < self.tokens_per_frame,  # I Frame
            self._bool_warp_where(
                (kv_idx < self.tokens_per_frame)
                | (
                    (self.num_frames * self.tokens_per_frame <= kv_idx)
                    & (
                        kv_idx
                        < self.IFrame_tokens + self.num_frames * self.tokens_per_frame
                    )
                ),
                True,
                False,
            ),
            self._bool_warp_where(
                q_idx < self.num_frames * self.tokens_per_frame,  # P Frame
                self._bool_warp_where(
                    kv_frame_idx < self.num_frames,
                    kv_frame_idx <= frame_idx,
                    self._bool_warp_where(
                        (kv_idx >= self.num_frames * self.tokens_per_frame)
                        & (
                            kv_idx
                            < self.num_frames * self.tokens_per_frame
                            + self.IFrame_tokens
                            + frame_idx * self.PFrame_tokens
                        ),
                        True,
                        False,
                    ),
                ),
                self._bool_warp_where(
                    q_idx
                    < self.num_frames * self.tokens_per_frame
                    + self.IFrame_tokens,  # I Frame tokens
                    self._bool_warp_where(
                        (kv_idx < self.tokens_per_frame)
                        | (
                            (self.num_frames * self.tokens_per_frame <= kv_idx)
                            & (
                                kv_idx
                                < self.IFrame_tokens
                                + self.num_frames * self.tokens_per_frame
                            )
                        ),
                        True,
                        False,
                    ),
                    self._bool_warp_where(
                        q_idx < self.seq_len,  # P Frame tokens
                        self._bool_warp_where(
                            (kv_idx < (token_frame_idx + 1) * self.tokens_per_frame)
                            | (
                                (kv_idx >= self.num_frames * self.tokens_per_frame)
                                & (
                                    kv_idx
                                    < self.num_frames * self.tokens_per_frame
                                    + self.IFrame_tokens
                                    + token_frame_idx * self.PFrame_tokens
                                )
                            ),
                            True,
                            False,
                        ),
                        False,
                    ),
                ),
            ),
        )

    def _mask_fn(self, b, h, q_idx, kv_idx):
        # I Frame
        if q_idx < self.tokens_per_frame:
            if kv_idx < self.tokens_per_frame or (
                self.num_frames * self.tokens_per_frame
                <= kv_idx
                < self.IFrame_tokens + self.num_frames * self.tokens_per_frame
            ):
                return True  # I frame can attend to self and I frame tokens
            else:
                return False
        elif q_idx < self.num_frames * self.tokens_per_frame:  # P frame
            frame_idx = q_idx // self.tokens_per_frame
            kv_frame_idx = kv_idx // self.tokens_per_frame
            if kv_frame_idx < self.num_frames:
                return kv_frame_idx <= frame_idx
            elif (
                kv_idx >= self.num_frames * self.tokens_per_frame
                and kv_idx
                < self.num_frames * self.tokens_per_frame
                + self.IFrame_tokens
                + frame_idx * self.PFrame_tokens
            ):
                return True
            else:
                return False
        elif (
            q_idx < self.num_frames * self.tokens_per_frame + self.IFrame_tokens
        ):  # I frame tokens
            if kv_idx < self.tokens_per_frame or (
                self.num_frames * self.tokens_per_frame
                <= kv_idx
                < self.IFrame_tokens + self.num_frames * self.tokens_per_frame
            ):
                return True
            else:
                return False
        elif q_idx < self.seq_len:  # P frame tokens
            token_frame_idx = (
                q_idx - self.IFrame_tokens - self.num_frames * self.tokens_per_frame
            ) // self.PFrame_tokens + 1
            if kv_idx < (token_frame_idx + 1) * self.tokens_per_frame or (
                kv_idx >= self.num_frames * self.tokens_per_frame
                and kv_idx
                < self.num_frames * self.tokens_per_frame
                + self.IFrame_tokens
                + token_frame_idx * self.PFrame_tokens
            ):
                return True
            else:
                return False
        else:
            return False
