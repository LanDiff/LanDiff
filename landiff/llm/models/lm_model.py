import logging
from typing import Any

import einops
import torch

from landiff.modules.packed_seq import PackedSeqlens
from landiff.utils import maybe_autocast, top_k_accuracy, top_p_probability

from ..modules.conditioner import MicroConditioner, TextCond
from ..modules.inference import KVCacheManager
from ..modules.tokenizer import SemanticEmbeddingModel, SemanticFrozenTokenizer
from ..modules.tokens import SequenceBuilder, Vocab
from .transformer import GPT, CondTransformerBase

logger = logging.getLogger(__name__)


class Semantic1DLM(CondTransformerBase):

    def __init__(
        self,
        tokenizer: SemanticFrozenTokenizer,
        cond_model: TextCond,
        transformer: GPT,
        micro_condition: MicroConditioner | None = None,
        train_with_packing: bool = True,
        use_chunked_cross_entropy=False,
        train2d=False,
        train3d=True,
        Iframe_len=330,
        Pframe_len=74,
        predict_motion_score=False,
        caculate_motion_socre_loss=False,
        fwd_dtype=torch.bfloat16,
        micro_cond_first=True,  # 是否micro_cond 在bos之后，早期的ar模型是在bos之后
        use_end_of_IFrame: bool = True,
        use_end_of_PFrame: bool = True,
    ):
        super().__init__(use_chunked_cross_entropy=use_chunked_cross_entropy)
        # learning components
        # condition model
        self.cond_model: TextCond = cond_model

        self.tokenizer: SemanticFrozenTokenizer = tokenizer
        self.transformer: GPT = transformer
        self.micro_condition: MicroConditioner | None = micro_condition

        # model config
        self.train_with_packing = train_with_packing
        self.Iframe_len = Iframe_len
        self.Pframe_len = Pframe_len
        self.train2d = train2d
        self.train3d = train3d
        self.predict_motion_score = predict_motion_score
        self.caculate_motion_socre_loss = caculate_motion_socre_loss
        self.fwd_dtype = fwd_dtype
        self.micro_cond_first = micro_cond_first
        self.use_end_of_IFrame = use_end_of_IFrame
        self.use_end_of_PFrame = use_end_of_PFrame
        if self.caculate_motion_socre_loss:
            assert (
                self.predict_motion_score
            ), "predict_motion_score must be True when caculate_motion_socre_loss is True"

        # prepare vocab
        self.vocab = Vocab()
        self.vocab.add_range("visual", self.tokenizer.vocab_size())
        # Keep a +1 to be compatible with an old checkpoint.
        # assert transformer.visual_vocab_size == self.tokenizer.vocab_size() + 3
        self.vocab.add_special("EOS")
        self.vocab.add_special("BOS")
        self.vocab.add_special("START_OF_IFrame")
        self.vocab.add_special("END_OF_IFrame")
        self.vocab.add_special("START_OF_PFrame")
        self.vocab.add_special("END_OF_PFrame")
        self.vocab.add_special("PAD")

        for sp in [
            "BOS",
            "START_OF_IFrame",
            "END_OF_IFrame",
            "START_OF_PFrame",
            "END_OF_PFrame",
            "EOS",
        ]:
            self.register_buffer(
                f"single_int_tensor_{sp}",
                torch.tensor([getattr(self.vocab, sp)]),
                persistent=False,
            )

        # visual embedding model, for image / video tokens embedding
        self.visual_embedding_model = SemanticEmbeddingModel(
            self.vocab.size(),
            transformer.hidden_dim,
        )
        if self.predict_motion_score:
            self.motion_score_predictor = torch.nn.Sequential(
                torch.nn.Linear(transformer.hidden_dim, transformer.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(transformer.hidden_dim, 1),
                torch.nn.ReLU(),  # make sure the score is positive
            )

    def _get_cond_embedding(self, x: Any):
        conditions = x.get("caption", x.get("class_id"))
        caption_embedding = x.get("caption_embedding")
        if caption_embedding is not None:
            return self.cond_model.forward_with_precomputed_embedding(caption_embedding)
        else:
            return self.cond_model(conditions)

    def _get_visual(self, inputs: dict) -> torch.Tensor:
        if self.train2d and self.train3d and "image" in inputs and "video" in inputs:
            visual = [
                x if x is not None else y
                for x, y in zip(inputs["image"], inputs["video"])
            ]
            visual = [
                einops.rearrange(x, "c h w -> 1 c h w") if x.ndim == 3 else x
                for x in visual
            ]
            if visual:
                return visual
        elif self.train3d and "video" in inputs:
            visual = [x for x in inputs["video"] if x is not None]
            if visual:
                visual = torch.stack(visual, dim=0)  # [B, T, C, H, W]
                return visual
        elif self.train2d and "image" in inputs:
            visual = [x for x in inputs["image"] if x is not None]
            if visual:
                visual = [einops.rearrange(x, "c h w -> 1 c h w") for x in visual]
                visual = torch.stack(visual, dim=0)
                return visual
        return None

    def faster_special_args(
        self, name: str
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        token_tensor: torch.LongTensor = getattr(self, f"single_int_tensor_{name}")
        token_id: int = getattr(self.vocab, name)
        return token_tensor, self.visual_embedding_model.lookup([token_id])

    def _get_visual_tokens(
        self, x: Any
    ) -> (
        tuple[list[torch.LongTensor], list[torch.FloatTensor], list[torch.LongTensor]]
        | tuple[None, None, None]
    ):
        codes = None
        visual: list[torch.Tensor] = self._get_visual(inputs=x)
        if codes is not None:
            pass
        elif visual is not None:
            with torch.no_grad():
                # Enable logs only in training, because it contains a worker synchronization.
                if self.train3d and self.train2d:
                    codes = []
                    for v in visual:
                        v = einops.rearrange(v, "t c h w -> 1 t c h w")
                        v_codes = self.tokenizer.encode_codes(v)
                        v_codes = [i.reshape(-1) for i in v_codes]
                        v_codes = torch.cat(v_codes, dim=0)
                        codes.append(v_codes)
                        # v = einops.rearrange(v, 't c h w -> 1 t c h w')
                        # code = self.tokenizer.encode_codes(v)  # encode uint8 TCHW tensor to vq indices
                        # code = code.reshape(-1)
                        # codes.append(code)
                else:
                    codes = self.tokenizer.encode_codes(visual)  # B L
                    codes = [einops.rearrange(c, "b ... -> b (...)") for c in codes]
                    codes = torch.cat(codes, dim=1)
        if codes is not None:
            if isinstance(codes, torch.Tensor):
                codes_embedding = self.visual_embedding_model(codes)  # B L D
            else:
                codes_embedding = []
                for code in codes:
                    code_embedding = self.visual_embedding_model(code)
                    codes_embedding.append(code_embedding)
            return codes, codes_embedding

        else:
            return None, None

    def tokenize(
        self, x: Any, with_guidance: bool = False
    ) -> tuple[list[SequenceBuilder], torch.Tensor | None, torch.Tensor | None]:
        # assert not ('image' in x and 'video' in x)
        assert not ("caption" in x and "class_id" in x)
        segment_length = self.tokenizer.segment_length
        Pframe_num = segment_length - 1
        # Performance note: run visual tokenizer before text encoder to achieve better performance.
        # This is due to high CPU overhead of the text encoder. Run it first will make the CPU busy and GPU idle.
        # For more details, see: https://dev.msh.team/train/Megatron-LM/-/merge_requests/501
        codes, codes_embedding = self._get_visual_tokens(x)
        # codes: list of 1D LongTensor, shaped (L,) where L = T * (H/stride_SAM/stride_VQ) * (W/stride_SAM/stride_VQ + 1)
        # codes_embedding: list of 2D FloatTensor, shaped (L, embed_dim)
        # visual_rope_indexs: 3D LongTensor meshgrid, shaped (L, 3)
        conditional_embedding = self._get_cond_embedding(x)
        micro_cond = None
        if self.micro_condition is not None:
            _, micro_cond = self.micro_condition(x)
        batch_size = len(conditional_embedding)

        if with_guidance:
            conditions = x.get("caption", x.get("class_id"))
            unconditional_embedding = self.cond_model.forward_unconditional(conditions)
            conds_embedding = [conditional_embedding, unconditional_embedding]
        else:
            conds_embedding = [conditional_embedding]
        # packing motion score label
        motion_score_label = None
        if self.caculate_motion_socre_loss:
            motion_score_label = x.get("motion_score")
            if motion_score_label is not None:
                motion_score_label = motion_score_label.reshape(-1).float()
        # do packing
        sequences: list[SequenceBuilder] = []
        motion_score_loss_masks = []
        for embedding in conds_embedding:
            for k in range(batch_size):  # TODO @keyu: hard to parallelize
                builder = SequenceBuilder()
                # add start bos
                builder.add_token_and_feature(
                    *self.faster_special_args("BOS"), loss_mask=False
                )
                if micro_cond is not None and self.micro_cond_first:
                    builder.add_feature(micro_cond[k], name="micro_cond")
                # add cond features
                builder.add_feature(embedding[k], name="cond")
                if micro_cond is not None and not self.micro_cond_first:
                    builder.add_feature(micro_cond[k], name="micro_cond")
                builder.add_token_and_feature(
                    *self.faster_special_args("START_OF_IFrame"), loss_mask=True
                )
                if codes is not None:
                    code_item = codes[k]
                    code_embedding_item = codes_embedding[k]
                    block_code_token_nums = (
                        self.Iframe_len + self.Pframe_len * Pframe_num
                    )
                    # assert (len(code_item) % block_code_token_nums -
                    #         self.Iframe_len) % self.Pframe_len == 0, f'Invalid codes length: {len(code_item)}'
                    for i in range(0, len(code_item), block_code_token_nums):
                        block_code_item = code_item[i : i + block_code_token_nums]
                        block_code_embedding_item = code_embedding_item[
                            i : i + block_code_token_nums
                        ]
                        block_index = i // block_code_token_nums
                        if i != 0:
                            builder.add_token_and_feature(
                                *self.faster_special_args("START_OF_IFrame"),
                                loss_mask=True,
                            )
                        builder.add_token_and_feature(
                            block_code_item[: self.Iframe_len],
                            block_code_embedding_item[: self.Iframe_len],
                            loss_mask=True,
                            name=f"Iframe_{block_index}",
                        )
                        if self.use_end_of_IFrame:
                            builder.add_token_and_feature(
                                *self.faster_special_args("END_OF_IFrame"),
                                loss_mask=True,
                            )
                        for j in range(
                            self.Iframe_len, len(block_code_item), self.Pframe_len
                        ):
                            builder.add_token_and_feature(
                                *self.faster_special_args("START_OF_PFrame"),
                                loss_mask=True,
                            )
                            builder.add_token_and_feature(
                                block_code_item[j : j + self.Pframe_len],
                                block_code_embedding_item[j : j + self.Pframe_len],
                                loss_mask=True,
                                name=f"Pframe_{block_index}_{(j-self.Iframe_len)//self.Pframe_len}",
                            )
                            if self.use_end_of_PFrame:
                                builder.add_token_and_feature(
                                    *self.faster_special_args("END_OF_PFrame"),
                                    loss_mask=True,
                                )

                # add eos
                builder.add_token_and_feature(
                    *self.faster_special_args("EOS"), loss_mask=True
                )
                sequences.append(builder)
                motion_score_loss_mask = torch.zeros(
                    len(builder), dtype=torch.bool, device=builder.device
                )
                motion_score_loss_mask[len(embedding[k])] = True
                motion_score_loss_masks.append(motion_score_loss_mask)
        motion_score_loss_mask = (
            torch.cat(motion_score_loss_masks, dim=0)
            if self.caculate_motion_socre_loss
            else None
        )
        return sequences, motion_score_label, motion_score_loss_mask

    def forward_packing(self, x):
        all_seqs, motion_score_label, motion_score_loss_mask = self.tokenize(x)
        seqlens = [len(x) for x in all_seqs]
        all_seqs = all_seqs[0].cat(all_seqs, keep_name=False)
        tokens = all_seqs.concat_tokens()  # [seqlen]
        embeddings = all_seqs.concat_features(None)  # [seqlen, dim]
        loss_mask = all_seqs.concat_loss_mask()  # [seqlen]
        if self.caculate_motion_socre_loss:
            assert len(loss_mask) == len(
                motion_score_loss_mask
            ), "loss_mask and motion_score_loss_mask must have the same length"

        # Shift sequence by 1 to perform next-token prediction.
        seqlens[-1] -= 1
        packed_seqlens = PackedSeqlens(seqlens)
        # For packing, the first token of every seq must have loss_mask=False.
        loss_mask[packed_seqlens.cu_seqlens(loss_mask.device)[1:-1]] = False
        labels, loss_mask = tokens[1:], loss_mask[1:]

        if not self.training:
            return (
                self.transformer(embeddings[:-1], seqlens=packed_seqlens),
                labels,
                loss_mask,
            )  # [tot_seqlen-1, vocab_size]

        if self.use_chunked_cross_entropy:
            hidden = self.transformer(
                embeddings[:-1], seqlens=packed_seqlens, apply_head=False
            )
            losses = self._chunked_losses(
                hidden, self.transformer.head, labels, loss_mask
            )
        else:
            hidden = self.transformer(
                embeddings[:-1], seqlens=packed_seqlens, apply_head=False
            )
            logits = self.transformer.hidden2logits(hidden)
            losses = self._losses(logits, labels, loss_mask)
        with torch.no_grad():
            eos_mask = labels == self.vocab.EOS
            eos_labels = labels[eos_mask]
            eos_logits = logits[eos_mask]
            eos_loss = torch.nn.functional.cross_entropy(eos_logits, eos_labels)
        if self.caculate_motion_socre_loss:
            # get motion score prediction hidden
            motion_score_loss_mask = motion_score_loss_mask[1:]
            motion_score_hidden = hidden[motion_score_loss_mask]
            assert len(motion_score_hidden) == len(
                motion_score_label
            ), "motion_score_hidden and motion_score_label must have the same length"
            with maybe_autocast(motion_score_hidden, self.fwd_dtype):
                pre_motion_score = self.motion_score_predictor(motion_score_hidden)
                pre_motion_score = pre_motion_score.reshape(-1).float()
                motion_score_label = motion_score_label.reshape(-1).float()
                assert len(pre_motion_score) == len(
                    motion_score_label
                ), "pre_motion_score and motion_score_label must have the same length"
                motion_score_loss = torch.nn.functional.mse_loss(
                    pre_motion_score, motion_score_label
                )
            losses["motion_score_loss"] = motion_score_loss

        # TODO: implement visualization
        return losses

    def video_frames_to_code_len(self, num_frames: int):
        segment_len = self.tokenizer.segment_length
        segment_stride = self.tokenizer.segment_stride
        res = 0
        for offset in range(0, num_frames, segment_stride):
            start_idx = offset
            end_idx = min(start_idx + segment_len, num_frames)
            frame_len = end_idx - start_idx
            res += self.Iframe_len + (frame_len - 1) * self.Pframe_len
            if self.use_end_of_IFrame and self.use_end_of_PFrame:
                res += 2 * frame_len
            else:
                res += frame_len
        return res

    @torch.no_grad()
    def sample(  # noqa C901
        self,
        inputs: dict[str, Any],
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        teacher_forcing: bool = False,
        num_frames: int = 13,
        use_gt_first_frame: bool = False,
        is_accuracy_calculated: bool = True,
        predict_eos: bool = False,
    ):
        """
        inputs: dict like the inputs to forward().
        guidance_scale: if > 0, will enable CF-guidance. The value should >1 to make sense.
        """
        Iframe_len = self.Iframe_len
        Pframe_len = self.Pframe_len
        segment_length = self.tokenizer.segment_length
        Pframe_num = segment_length - 1
        with_guidance = guidance_scale > 0 and guidance_scale != 1
        assert (
            top_k is None or top_p is None
        ), "Top_k and Top_p can not exist at the same time."
        all_seqs, motion_score_label, motion_score_loss_mask = self.tokenize(
            inputs, with_guidance=with_guidance
        )
        all_seqs: list[SequenceBuilder]
        batch_size = len(all_seqs) // 2 if with_guidance else len(all_seqs)

        # 简单假设输入的batch size为1,同时填充了gt的index和feature
        tokens, gt_features, loss_mask = SequenceBuilder.batch(all_seqs)
        token = tokens[0]
        start_of_iframe_index = 0

        # 查询token 中vocab.START_OF_VISUAL的位置
        start_of_iframe_index = (token == self.vocab.START_OF_IFrame).nonzero(
            as_tuple=True
        )
        # assert len(start_of_iframe_index) == 1, f'Invalid start_of_visual shape: {start_of_iframe_index.shape}'
        start_of_iframe_index = start_of_iframe_index[0].cpu().tolist()
        start_of_iframe_index = sorted(start_of_iframe_index)
        start_of_iframe_index = int(start_of_iframe_index[0])

        if not use_gt_first_frame:
            start_of_visual = start_of_iframe_index
        else:
            if self.use_end_of_IFrame:
                start_of_visual = start_of_iframe_index + Iframe_len + 2
            else:
                start_of_visual = start_of_iframe_index + Iframe_len + 1
        prefix_len: int = start_of_visual + 1
        # 切割feature和rope indexs到visual_start
        features = gt_features[:, :prefix_len]

        full_len = start_of_iframe_index  # prefix condition部分
        full_len += self.video_frames_to_code_len(num_frames)
        full_len += 1  # EOS
        if not use_gt_first_frame:
            sampled_codes = []
        else:
            sampled_codes = [
                token[
                    start_of_iframe_index + 1 : start_of_iframe_index + 1 + Iframe_len
                ].reshape(batch_size, -1)
            ]
        start_of_iframe_set = set()
        iframe_set = set()
        pframe_set = set()
        end_of_iframe_set = set()
        start_of_pframe_set = set()
        end_of_pframe_set = set()
        possible_eos_token_set = (
            set()
        )  # 可能的eos位置, 用于判断是否需要提前结束,位置为frame end token的后面

        if self.use_end_of_IFrame and self.use_end_of_PFrame:
            visual_block_len = (
                Iframe_len + Pframe_num * Pframe_len + segment_length * 2
            )  # visual frame + 边界token
        else:
            visual_block_len = (
                Iframe_len + Pframe_num * Pframe_len + segment_length
            )  # visual frame + 边界token
        logger_seq = False
        for index in range(start_of_iframe_index, full_len - 1, visual_block_len):
            move_index = index
            start_of_iframe_set.add(move_index)
            if logger_seq:
                print("SI", end=" ")
            move_index += 1
            iframe_set.update(range(move_index + 1, move_index + 1 + Iframe_len))
            if logger_seq:
                print(" ".join(["*"] * Iframe_len), end=" ")
            move_index += Iframe_len
            if self.use_end_of_IFrame:
                end_of_iframe_set.add(move_index)
                if logger_seq:
                    print("EI", end=" ")
                move_index += 1
            if index > start_of_iframe_index:  # 第一个visual block不需要添加
                possible_eos_token_set.add(move_index)
            if self.use_end_of_IFrame and self.use_end_of_PFrame:
                p_frame_end = min(
                    full_len - 1,
                    move_index - 1 + Pframe_len * Pframe_num + 2 * Pframe_num,
                )
                j_generater = range(move_index, p_frame_end, Pframe_len + 2)
            else:
                p_frame_end = min(
                    full_len - 1, move_index - 1 + Pframe_len * Pframe_num + Pframe_num
                )
                j_generater = range(move_index, p_frame_end, Pframe_len + 1)
            for j in j_generater:
                start_of_pframe_set.add(j)
                if logger_seq:
                    print("SP", end=" ")
                move_index += 1
                pframe_set.update(range(j + 1, j + 1 + Pframe_len))
                if logger_seq:
                    print(" ".join(["*"] * Pframe_len), end=" ")
                move_index += Pframe_len
                if self.use_end_of_PFrame:
                    end_of_pframe_set.add(j + Pframe_len + 1)
                    if logger_seq:
                        print("EP", end=" ")
                    move_index += 1
                if index > start_of_iframe_index:  # 第一个visual block不需要添加
                    possible_eos_token_set.add(move_index)

        if seed:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
        else:
            generator = None

        last_code = None
        freqs_cis = self.transformer.rope.get_freqs_cis_by_seqlens([full_len])
        freqs_cis = einops.rearrange(freqs_cis, "t c -> 1 t c")
        infer_freqs_cis = freqs_cis[:, :prefix_len]
        with KVCacheManager(self):
            for i in range(prefix_len, full_len):
                # Otherwise, use feature from last step
                if last_code is not None:
                    features = self.visual_embedding_model(last_code)
                    infer_freqs_cis = freqs_cis[:, i - 1 : i]  # rope index
                    if with_guidance:
                        features = torch.cat([features, features], dim=0)
                if with_guidance:
                    infer_freqs_cis = torch.cat(
                        [infer_freqs_cis, infer_freqs_cis], dim=0
                    )

                logits_ = self.transformer.sample(
                    features,
                    prefix_paddings=None,
                    freqs_cis=infer_freqs_cis,
                )
                logits_ = logits_.to(dtype=torch.float32)
                if with_guidance:
                    logits_, uncond_logits_ = torch.split(logits_, batch_size, dim=0)
                    logits_ = uncond_logits_ + guidance_scale * (
                        logits_ - uncond_logits_
                    )

                full_logits = logits_.clone()
                logits_ = logits_ / temperature
                possable_indexes = []
                if i in start_of_iframe_set:
                    possable_indexes.append(self.vocab.START_OF_IFrame)
                if i in start_of_pframe_set:
                    possable_indexes.append(self.vocab.START_OF_PFrame)
                if i in possible_eos_token_set:
                    possable_indexes.append(self.vocab.EOS)
                if len(possable_indexes) == 0:
                    if top_k is not None:
                        v, ix = torch.topk(logits_, top_k)
                        logits_[logits_ < v[:, [-1]]] = -float("Inf")
                    probs = torch.nn.functional.softmax(logits_, dim=-1)

                    if top_p is not None:
                        probs = top_p_probability(top_p, probs)
                else:
                    # 将logits中不在possable_indexes中的值设为负无穷
                    mask = torch.full_like(logits_, fill_value=-float("Inf"))
                    mask[0, possable_indexes] = 0
                    logits_ = logits_ + mask
                    probs = torch.nn.functional.softmax(logits_, dim=-1)

                last_code = torch.multinomial(probs, num_samples=1, generator=generator)
                if (
                    i in possible_eos_token_set
                    and predict_eos
                    and int(last_code.item()) == int(self.vocab.EOS)
                ):
                    # 判断last_code是否为eos
                    logger.info(f"Predicted EOS at {i-prefix_len} token")
                    print(f"Predicted EOS at {i-prefix_len} token")
                    break
                elif i in start_of_iframe_set:
                    last_code = (
                        torch.tensor(
                            [self.vocab.START_OF_IFrame], device=last_code.device
                        )
                        .long()
                        .reshape(last_code.shape)
                    )
                    continue
                elif i in end_of_iframe_set:
                    last_code = (
                        torch.tensor(
                            [self.vocab.END_OF_IFrame], device=last_code.device
                        )
                        .long()
                        .reshape(last_code.shape)
                    )
                    continue
                elif i in start_of_pframe_set:
                    last_code = (
                        torch.tensor(
                            [self.vocab.START_OF_PFrame], device=last_code.device
                        )
                        .long()
                        .reshape(last_code.shape)
                    )
                    continue
                elif i in end_of_pframe_set:
                    last_code = (
                        torch.tensor(
                            [self.vocab.END_OF_PFrame], device=last_code.device
                        )
                        .long()
                        .reshape(last_code.shape)
                    )
                    continue
                elif i == full_len - 1:
                    last_code = (
                        torch.tensor([self.vocab.EOS], device=last_code.device)
                        .long()
                        .reshape(last_code.shape)
                    )
                    continue
                # print(f'topk_acc: {topk_acc}')
                sampled_codes.append(last_code)
                if teacher_forcing:
                    last_code = token[i : i + 1].reshape(last_code.shape)
        sampled_codes = torch.cat(sampled_codes, dim=1)
        visual_token_num = len(iframe_set) + len(pframe_set)
        if not predict_eos:
            assert (
                sampled_codes.shape[1] == visual_token_num
            ), f"Invalid sampled_codes shape: {sampled_codes.shape} != visual_token_num"
        sampled_codes.clamp_(min=0, max=self.tokenizer.vocab_size() - 1)
        return sampled_codes
