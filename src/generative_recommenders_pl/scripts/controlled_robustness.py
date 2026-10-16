"""Stage-two perturbations and metrics, shared by the runner and audit tests."""

from dataclasses import dataclass

import numpy as np
import torch

from generative_recommenders_pl.models.utils.inference import seen_item_mask
from generative_recommenders_pl.scripts.run_robustness import (
    Perturbation, RecommendationOutput, _clone_sample, _make_perturbations,
    _unique_position_indices, _jensen_shannon_divergence, _topk_overlap, _jaccard,
    _rank_delta,
)

TIMESTAMP_SHIFTS = (-604800, -86400, -3600, 3600, 86400, 604800)


@dataclass(frozen=True)
class ControlledPerturbation:
    perturbation: Perturbation
    requested_shift: int = 0
    actual_shift: int = 0
    timestamp_changed: bool = False
    changed_time_buckets: int = 0


def extended_positions(length: int, tail_count: int = 10):
    positions = _unique_position_indices(length)
    seen = {index for _, index in positions}
    for distance in range(min(length, tail_count)):
        index = length - 1 - distance
        if index not in seen:
            positions.append((f"tail_{distance}", index))
            seen.add(index)
    return positions


def shift_timestamp(sample, index: int, seconds: int):
    length = int(sample["history_lengths"])
    if not 0 <= index < length:
        raise IndexError("Timestamp index outside the real history")
    perturbed = _clone_sample(sample)
    times = sample["historical_timestamps"]
    before = int(times[index])
    lower = int(times[index - 1]) if index else 0
    upper = int(times[index + 1]) if index + 1 < length else int(sample["target_timestamps"])
    if not lower <= before <= upper:
        raise ValueError("Timestamp perturbations require chronological input")
    after = min(upper, max(lower, before + seconds))
    perturbed["historical_timestamps"][index] = after
    return perturbed, after - before


def _buckets(delta):
    return np.clip((np.log(np.maximum(np.abs(delta), 1).astype(np.float32)) / 0.301).astype(np.int64), 0, 128)


def changed_time_bucket_count(sample, index, new_timestamp):
    """Count changed active causal time-bias entries in O(L), not O(L^2).

    Bias entry (q,k) uses timestamp[q+1]-timestamp[k]. Changing t_i
    affects column i for q>=i, and row i-1 for k<=i-1. These causal
    regions do not intersect. The target timestamp is included at t_L.
    """
    length = int(sample["history_lengths"])
    times = np.append(sample["historical_timestamps"][:length].numpy(), int(sample["target_timestamps"]))
    old = int(times[index])
    column_before = _buckets(times[index + 1:] - old)
    column_after = _buckets(times[index + 1:] - new_timestamp)
    changed = int(np.count_nonzero(column_before != column_after))
    if index:
        changed += int(np.count_nonzero(
            _buckets(old - times[:index]) != _buckets(new_timestamp - times[:index])
        ))
    return changed


def make_controlled_perturbations(sample, all_item_ids, repeats, seed, user_index, tail_count=10, timestamp_shifts=TIMESTAMP_SHIFTS):
    length = int(sample["history_lengths"])
    if length <= 1:
        raise ValueError("Controlled deletion requires at least two real history items")
    positions = extended_positions(length, tail_count)
    legacy = _make_perturbations(sample, all_item_ids, repeats, seed, user_index, positions=positions)
    result = [ControlledPerturbation(p) for p in legacy]
    for label, index in positions:
        for repetition, seconds in enumerate(timestamp_shifts):
            shifted, actual = shift_timestamp(sample, index, seconds)
            item = int(sample["historical_ids"][index])
            p = Perturbation(shifted, "timestamp_only", label, index, item, item, repetition, None)
            result.append(ControlledPerturbation(
                p, int(seconds), actual, actual != 0,
                changed_time_bucket_count(sample, index, int(shifted["historical_timestamps"][index])),
            ))
    return result


def recommendations_from_logits(raw_logits, candidate_ids, mask, target_ids, top_k):
    if top_k < 1 or ((~mask).sum(dim=1) < top_k).any():
        raise ValueError("Every sample needs at least top_k unmasked candidates")
    logits = raw_logits.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(logits, dim=1)
    scores, offsets = torch.topk(logits, top_k, dim=1, sorted=True)
    ids = candidate_ids[offsets]
    logits, probabilities, scores, ids = [x.detach().cpu() for x in (logits, probabilities, scores, ids)]
    candidate_ids = candidate_ids.cpu()
    outputs = []
    for row, target in enumerate(target_ids):
        match = (candidate_ids == target).nonzero(as_tuple=True)[0]
        rank = None
        if match.numel():
            value = logits[row, int(match[0])]
            if torch.isfinite(value):
                rank = int((logits[row] > value).sum()) + 1
        outputs.append(RecommendationOutput(ids[row].tolist(), scores[row].tolist(), probabilities[row], rank))
    return outputs


def comparison_metrics(original, perturbed, top_k, target_id):
    return {
        "topk_overlap": _topk_overlap(original.top_ids, perturbed.top_ids, top_k),
        "jaccard": _jaccard(original.top_ids, perturbed.top_ids, top_k),
        "js_divergence": _jensen_shannon_divergence(original.probabilities, perturbed.probabilities),
        "original_top1_id": original.top_ids[0],
        "perturbed_top1_id": perturbed.top_ids[0],
        "top1_changed": original.top_ids[0] != perturbed.top_ids[0],
        "original_target_rank": original.target_rank,
        "perturbed_target_rank": perturbed.target_rank,
        "target_rank_delta": _rank_delta(original.target_rank, perturbed.target_rank),
        "original_target_in_topk": target_id in original.top_ids[:top_k],
        "perturbed_target_in_topk": target_id in perturbed.top_ids[:top_k],
    }
