import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import hydra
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from torch.utils.data._utils.collate import default_collate

from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import seq_features_from_row
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")

POSITION_SPECS = (
    ("first", 0.0),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("last", 1.0),
)


@dataclass(frozen=True)
class RecommendationOutput:
    top_ids: list[int]
    top_scores: list[float]
    probabilities: torch.Tensor
    target_rank: int | None


@dataclass(frozen=True)
class Perturbation:
    sample: dict[str, Any]
    perturbation_type: str
    position_label: str
    position_index: int
    original_item_id: int
    perturbed_item_id: int | None
    repetition: int
    seed: int | None


def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    try:
        value = cfg.get(key, default)
    except MissingMandatoryValue:
        return default
    if value is None or value == "???":
        return default
    return value


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain a state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=False)


def _unique_position_indices(history_length: int) -> list[tuple[str, int]]:
    if history_length <= 0:
        return []

    positions: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, fraction in POSITION_SPECS:
        index = round((history_length - 1) * fraction)
        index = max(0, min(history_length - 1, index))
        if index not in seen:
            seen.add(index)
            positions.append((label, index))
    return positions


def _replace_item(
    sample: dict[str, Any],
    position_index: int,
    replacement_item_id: int,
) -> dict[str, Any]:
    perturbed = _clone_sample(sample)
    perturbed["historical_ids"][position_index] = replacement_item_id
    return perturbed


def _delete_item(sample: dict[str, Any], position_index: int) -> dict[str, Any]:
    perturbed = _clone_sample(sample)
    history_length = int(perturbed["history_lengths"])
    if position_index < 0 or position_index >= history_length:
        raise IndexError(
            f"position_index={position_index} is outside history length {history_length}"
        )

    for key in ("historical_ids", "historical_ratings", "historical_timestamps"):
        values = perturbed[key].clone()
        kept = torch.cat(
            [
                values[:position_index],
                values[position_index + 1 : history_length],
            ],
            dim=0,
        )
        values.zero_()
        if kept.numel() > 0:
            values[: kept.numel()] = kept
        perturbed[key] = values

    perturbed["history_lengths"] = history_length - 1
    return perturbed


def _make_perturbations(
    sample: dict[str, Any],
    all_item_ids: list[int],
    replacement_repeats: int,
    base_seed: int,
    sample_index: int,
) -> list[Perturbation]:
    history_length = int(sample["history_lengths"])
    forbidden_item_ids = {
        int(item_id)
        for item_id in sample["historical_ids"][:history_length].tolist()
    }
    forbidden_item_ids.add(int(sample["target_ids"]))
    replacement_pool = [
        candidate for candidate in all_item_ids if candidate not in forbidden_item_ids
    ]
    perturbations: list[Perturbation] = []

    for position_label, position_index in _unique_position_indices(history_length):
        original_item_id = int(sample["historical_ids"][position_index].item())

        delete_sample = _delete_item(sample, position_index)
        perturbations.append(
            Perturbation(
                sample=delete_sample,
                perturbation_type="delete",
                position_label=position_label,
                position_index=position_index,
                original_item_id=original_item_id,
                perturbed_item_id=None,
                repetition=0,
                seed=None,
            )
        )

        for repetition in range(replacement_repeats):
            seed = base_seed + sample_index * 1009 + position_index * 31 + repetition
            generator = torch.Generator()
            generator.manual_seed(seed)
            offset = int(
                torch.randint(
                    low=0,
                    high=len(replacement_pool),
                    size=(1,),
                    generator=generator,
                ).item()
            )
            replacement_item_id = int(replacement_pool[offset])
            replacement_sample = _replace_item(
                sample,
                position_index=position_index,
                replacement_item_id=replacement_item_id,
            )
            perturbations.append(
                Perturbation(
                    sample=replacement_sample,
                    perturbation_type="replace_random",
                    position_label=position_label,
                    position_index=position_index,
                    original_item_id=original_item_id,
                    perturbed_item_id=replacement_item_id,
                    repetition=repetition,
                    seed=seed,
                )
            )

    return perturbations


def _batch_samples(samples: Iterable[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return default_collate(list(samples))


def _score_samples(
    model,
    samples: list[dict[str, Any]],
    candidate_ids: torch.Tensor,
    candidate_embeddings_t: torch.Tensor,
    target_ids: list[int],
    top_k: int,
    device: torch.device,
) -> list[RecommendationOutput]:
    batch = _batch_samples(samples)
    seq_features, _, _ = seq_features_from_row(
        batch,
        device=device,
        max_output_length=model.gr_output_length + 1,
    )
    input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
    seq_features = seq_features._replace(past_embeddings=input_embeddings)

    seq_embeddings, _ = model.forward(seq_features)
    current_embeddings = ops.get_current_embeddings(
        seq_features.past_lengths,
        seq_embeddings,
    )
    logits = torch.mm(current_embeddings, candidate_embeddings_t)

    invalid_mask = (
        candidate_ids.unsqueeze(0).unsqueeze(2)
        == seq_features.past_ids.unsqueeze(1)
    ).any(dim=2)
    logits = logits.masked_fill(invalid_mask, float("-inf"))
    probabilities = F.softmax(logits, dim=1)

    actual_top_k = min(top_k, candidate_ids.numel())
    top_scores, top_offsets = torch.topk(
        logits,
        k=actual_top_k,
        dim=1,
        sorted=True,
    )
    top_ids = candidate_ids[top_offsets]

    outputs: list[RecommendationOutput] = []
    for row_idx, target_id in enumerate(target_ids):
        target_matches = (candidate_ids == int(target_id)).nonzero(as_tuple=True)[0]
        target_rank: int | None = None
        if target_matches.numel() > 0:
            target_offset = int(target_matches[0].item())
            target_score = logits[row_idx, target_offset]
            if torch.isfinite(target_score):
                target_rank = int((logits[row_idx] > target_score).sum().item() + 1)

        outputs.append(
            RecommendationOutput(
                top_ids=[int(x) for x in top_ids[row_idx].detach().cpu().tolist()],
                top_scores=[
                    float(x) for x in top_scores[row_idx].detach().cpu().tolist()
                ],
                probabilities=probabilities[row_idx].detach().cpu(),
                target_rank=target_rank,
            )
        )
    return outputs


def _jensen_shannon_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    p = p.to(torch.float64)
    q = q.to(torch.float64)
    m = 0.5 * (p + q)

    def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return (a[mask] * ((a[mask] + eps).log() - (b[mask] + eps).log())).sum()

    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def _topk_overlap(original_ids: list[int], perturbed_ids: list[int], top_k: int) -> float:
    original_set = set(original_ids[:top_k])
    perturbed_set = set(perturbed_ids[:top_k])
    return len(original_set & perturbed_set) / float(top_k)


def _jaccard(original_ids: list[int], perturbed_ids: list[int], top_k: int) -> float:
    original_set = set(original_ids[:top_k])
    perturbed_set = set(perturbed_ids[:top_k])
    union = original_set | perturbed_set
    return len(original_set & perturbed_set) / float(len(union)) if union else 0.0


def _rank_delta(original_rank: int | None, perturbed_rank: int | None) -> int | None:
    if original_rank is None or perturbed_rank is None:
        return None
    return perturbed_rank - original_rank


def _write_results(rows: list[dict[str, Any]], output_file: str) -> None:
    output_path = Path(to_absolute_path(output_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    log.info(f"Saved {len(frame)} robustness rows to {output_path}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt_path is None or cfg.ckpt_path == "":
        raise ValueError("Please provide ckpt_path=... for robustness evaluation")

    output_file = str(
        _cfg_get(cfg, "output_file", "outputs/robustness/ml1m_hstu_robustness.csv")
    )
    num_users = int(_cfg_get(cfg, "num_users", 500))
    user_offset = int(_cfg_get(cfg, "user_offset", 0))
    replacement_repeats = int(_cfg_get(cfg, "replacement_repeats", 5))
    base_seed = int(_cfg_get(cfg, "base_seed", 42))
    top_k = int(_cfg_get(cfg, "top_k", 10))
    progress_every = int(_cfg_get(cfg, "progress_every", 25))
    device_name = _cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.setup("predict")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        datamodule=datamodule,
        _recursive_=False,
    )
    _load_checkpoint(model, cfg.ckpt_path)
    model.to(device)
    model.eval()

    candidate_ids = torch.tensor(datamodule.all_item_ids, dtype=torch.long, device=device)
    candidate_embeddings = model.negatives_sampler.normalize_embeddings(
        model.embeddings.get_item_embeddings(candidate_ids)
    )
    candidate_embeddings_t = candidate_embeddings.t().contiguous()

    dataset = datamodule.test_dataset
    end_index = min(user_offset + num_users, len(dataset))
    if user_offset < 0 or user_offset >= len(dataset):
        raise IndexError(
            f"user_offset={user_offset} is outside dataset range [0, {len(dataset) - 1}]"
        )
    if end_index - user_offset < num_users:
        log.warning(
            f"Requested {num_users} users from offset {user_offset}, "
            f"but dataset only has {end_index - user_offset} available users."
        )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for relative_idx, sample_index in enumerate(range(user_offset, end_index), start=1):
            sample = dataset[sample_index]
            history_length = int(sample["history_lengths"])
            if history_length <= 0:
                continue

            perturbations = _make_perturbations(
                sample=sample,
                all_item_ids=datamodule.all_item_ids,
                replacement_repeats=replacement_repeats,
                base_seed=base_seed,
                sample_index=sample_index,
            )

            all_samples = [sample] + [perturbation.sample for perturbation in perturbations]
            target_id = int(sample["target_ids"])
            target_ids = [target_id] * len(all_samples)
            outputs = _score_samples(
                model=model,
                samples=all_samples,
                candidate_ids=candidate_ids,
                candidate_embeddings_t=candidate_embeddings_t,
                target_ids=target_ids,
                top_k=max(top_k, 1),
                device=device,
            )

            original_output = outputs[0]
            for perturbation, perturbed_output in zip(perturbations, outputs[1:]):
                rows.append(
                    {
                        "user_index": sample_index,
                        "user_id": int(sample["user_id"]),
                        "history_length": history_length,
                        "target_id": target_id,
                        "target_rating": int(sample["target_ratings"]),
                        "perturbation_type": perturbation.perturbation_type,
                        "position_label": perturbation.position_label,
                        "position_index": perturbation.position_index,
                        "position_fraction": (
                            perturbation.position_index / (history_length - 1)
                            if history_length > 1
                            else 0.0
                        ),
                        "original_item_id": perturbation.original_item_id,
                        "perturbed_item_id": perturbation.perturbed_item_id,
                        "repetition": perturbation.repetition,
                        "seed": perturbation.seed,
                        "top_k": top_k,
                        "topk_overlap": _topk_overlap(
                            original_output.top_ids,
                            perturbed_output.top_ids,
                            top_k,
                        ),
                        "jaccard": _jaccard(
                            original_output.top_ids,
                            perturbed_output.top_ids,
                            top_k,
                        ),
                        "js_divergence": _jensen_shannon_divergence(
                            original_output.probabilities,
                            perturbed_output.probabilities,
                        ),
                        "original_top1_id": original_output.top_ids[0],
                        "perturbed_top1_id": perturbed_output.top_ids[0],
                        "top1_changed": original_output.top_ids[0]
                        != perturbed_output.top_ids[0],
                        "original_target_rank": original_output.target_rank,
                        "perturbed_target_rank": perturbed_output.target_rank,
                        "target_rank_delta": _rank_delta(
                            original_output.target_rank,
                            perturbed_output.target_rank,
                        ),
                        "original_target_in_topk": target_id
                        in set(original_output.top_ids[:top_k]),
                        "perturbed_target_in_topk": target_id
                        in set(perturbed_output.top_ids[:top_k]),
                        "original_topk_ids": json.dumps(original_output.top_ids[:top_k]),
                        "perturbed_topk_ids": json.dumps(perturbed_output.top_ids[:top_k]),
                    }
                )

            if progress_every > 0 and relative_idx % progress_every == 0:
                log.info(
                    f"Processed {relative_idx}/{end_index - user_offset} users; "
                    f"rows={len(rows)}"
                )

    _write_results(rows, output_file)


if __name__ == "__main__":
    main()
