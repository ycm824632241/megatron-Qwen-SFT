"""Apply target-preserving left truncation to Bridge GPT chat SFT samples.

Bridge's legacy ``GPTSFTChatDataset`` tokenizes the complete conversation and
later truncates ``input_ids`` and ``loss_mask`` with ``[:max_seq_length]``.
For ``loss_mode=last_turn`` this can remove the entire supervised assistant
span.  Truncating the processed example before collation also covers offline
packing, whose histogram otherwise skips over-length examples.
"""

from __future__ import annotations

import functools
import os
from typing import Any


_PATCHED = False


def left_truncate_processed_example(processed: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    """Keep the final ``max_seq_length + 1`` tokens and aligned SFT fields.

    One extra stored token is required because GPT inputs/labels are shifted by
    one position during collation and offline packing.
    """
    input_ids = processed.get("input_ids")
    if input_ids is None:
        return processed

    stored_limit = int(max_seq_length) + 1
    input_length = len(input_ids)
    if input_length <= stored_limit:
        return processed

    start = input_length - stored_limit
    original_context_length = len(processed.get("context_ids", ()))
    truncated_input = input_ids[start:]
    processed["input_ids"] = truncated_input

    loss_mask = processed.get("loss_mask")
    if loss_mask is not None:
        processed["loss_mask"] = loss_mask[start:]

    # Rebuild context/answer from the same retained token window so metadata,
    # generation evaluation, and packed training all agree on the boundary.
    #
    # input_ids keeps seq+1 for the GPT shift; context/answer are NOT shifted
    # in collate and must be <= max_seq_length, otherwise:
    #   ValueError: expected sequence of length 8193 at dim 1 (got 8192)
    new_context_length = max(0, min(stored_limit, original_context_length - start))
    context_keep = min(new_context_length, int(max_seq_length))
    if "context_ids" in processed:
        processed["context_ids"] = truncated_input[:context_keep]
    if "answer_ids" in processed:
        processed["answer_ids"] = truncated_input[new_context_length : new_context_length + int(max_seq_length)]
    if "context_length" in processed:
        processed["context_length"] = context_keep
    if "answer_start_idx" in processed:
        processed["answer_start_idx"] = context_keep
    if "token_count" in processed:
        processed["token_count"] = len(truncated_input)
    return processed


def apply() -> str:
    """Patch Bridge only when ``QWEN35_SFT_TRUNCATION_SIDE=left``."""
    global _PATCHED
    side = os.environ.get("QWEN35_SFT_TRUNCATION_SIDE", "right").lower()
    if side != "left":
        return f"disabled ({side})"
    if _PATCHED:
        return "already applied"

    from megatron.bridge.data.datasets.gpt_sft import GPTSFTChatDataset

    original = GPTSFTChatDataset._process_example

    @functools.wraps(original)
    def wrapped(self, example):
        processed = original(self, example)
        return left_truncate_processed_example(processed, self.max_seq_length)

    GPTSFTChatDataset._process_example = wrapped

    # collate only pads; a seq+1 answer would stay 8193 while others pad to 8192.
    original_collate_item = GPTSFTChatDataset._collate_item

    @functools.wraps(original_collate_item)
    def collate_item(self, item, max_length, pad_id):
        clipped = [list(x)[:max_length] for x in item]
        return original_collate_item(self, clipped, max_length, pad_id)

    GPTSFTChatDataset._collate_item = collate_item
    _PATCHED = True
    return "applied (keep tail; aligned input/loss/context/answer)"
