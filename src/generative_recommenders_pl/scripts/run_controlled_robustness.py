"""Fixed/dynamic candidate masks, time-only shifts and last-ten-position study."""

import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from generative_recommenders_pl.models.utils.inference import logits_from_features, seen_item_mask
from generative_recommenders_pl.scripts.audit_robustness import audit_model, features_for
from generative_recommenders_pl.scripts.controlled_robustness import (
    TIMESTAMP_SHIFTS, make_controlled_perturbations, recommendations_from_logits, comparison_metrics,
)
from generative_recommenders_pl.scripts.run_robustness import _cfg_get, _load_checkpoint, _unique_position_indices


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval.yaml")
def main(cfg):
    output_dir = Path(to_absolute_path(str(_cfg_get(cfg, "stage2_output_dir", "outputs/robustness/stage2"))))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "controlled_results.csv"
    if result_path.exists() and not bool(_cfg_get(cfg, "overwrite", False)):
        raise FileExistsError(f"{result_path} exists; use another output directory or +overwrite=true")
    root = Path(to_absolute_path("."))
    checkpoint = Path(to_absolute_path(str(cfg.ckpt_path)))
    num_users = int(_cfg_get(cfg, "num_users", 500))
    offset = int(_cfg_get(cfg, "user_offset", 0))
    repeats = int(_cfg_get(cfg, "replacement_repeats", 5))
    seed = int(_cfg_get(cfg, "base_seed", 42))
    top_k = int(_cfg_get(cfg, "top_k", 10))
    batch_size = int(_cfg_get(cfg, "inference_batch_size", 32))
    tail_count = int(_cfg_get(cfg, "tail_count", 10))
    shifts = tuple(int(s) for s in _cfg_get(cfg, "timestamp_shifts", TIMESTAMP_SHIFTS))
    if min(num_users, repeats, top_k, batch_size, tail_count) <= 0 or offset < 0:
        raise ValueError("Invalid protocol sizes or user offset")
    device = torch.device(_cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.setup("predict")
    dataset = datamodule.test_dataset
    if offset + num_users > len(dataset):
        raise ValueError("The full requested user cohort is unavailable")
    indices = list(range(offset, offset + num_users))
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_names = set(model.state_dict())
    missing = sorted(trained_names - set(checkpoint_data["state_dict"]))
    if missing:
        raise ValueError(f"Checkpoint is missing model weights: {missing}")
    _load_checkpoint(model, str(checkpoint))
    model.to(device).eval()
    with torch.no_grad():
        candidate_ids = torch.tensor(datamodule.all_item_ids, dtype=torch.long, device=device)
        candidates_t = model.negatives_sampler.normalize_embeddings(model.embeddings.get_item_embeddings(candidate_ids)).t().contiguous()

    cohort = []
    for index in indices:
        sample = dataset[index]
        length = int(sample["history_lengths"])
        if length <= 1:
            raise ValueError(f"User {index} has too short a history for deletion; no silent cohort changes")
        historical = {key: sample[key][:length].tolist() for key in ("historical_ids", "historical_ratings", "historical_timestamps")}
        historical.update(target_id=int(sample["target_ids"]), target_timestamp=int(sample["target_timestamps"]))
        cohort.append({"user_index": index, "user_id": int(sample["user_id"]), "history_length": length,
                       "input_sha256": hashlib.sha256(json.dumps(historical, sort_keys=True).encode()).hexdigest()})
    source_files = ["src/generative_recommenders_pl/scripts/run_robustness.py", "src/generative_recommenders_pl/scripts/controlled_robustness.py", "src/generative_recommenders_pl/scripts/run_controlled_robustness.py", "src/generative_recommenders_pl/scripts/audit_robustness.py", "src/generative_recommenders_pl/models/utils/inference.py", "src/generative_recommenders_pl/models/utils/features.py", "src/generative_recommenders_pl/models/sequential_encoders/hstu.py"]
    manifest = {
        "status": "running", "checkpoint": str(checkpoint.relative_to(root)) if checkpoint.is_relative_to(root) else str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_epoch": checkpoint_data.get("epoch"),
        "protocol": {"num_users": num_users, "user_offset": offset, "replacement_repeats": repeats, "base_seed": seed, "top_k": top_k,
                     "seed_formula": "base_seed + user_index * 1009 + position_index * 31 + repetition",
                     "tail_count": tail_count, "timestamp_shifts_seconds": shifts,
                     "timestamp_clipping": "previous <= changed <= next; first lower bound 0; last upper bound target timestamp",
                     "mask_modes": ["dynamic", "fixed_original"], "timestamp_target": "unchanged; prediction-time anchor at current length",
                     "replacement_pool": "catalogue order; excludes original history and target"},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inference_batch_size": batch_size, "cohort": cohort,
        "candidate_ids_sha256": hashlib.sha256(json.dumps(datamodule.all_item_ids).encode()).hexdigest(),
        "input_files_sha256": {p: file_sha256(root / p) for p in ("tmp/ml-1m/sasrec_format.csv", "tmp/processed/ml-1m/movies.csv")},
        "source_files_sha256": {p: file_sha256(root / p) for p in source_files},
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    audit_count = min(int(_cfg_get(cfg, "audit_users", 12)), num_users)
    if audit_count <= 0:
        raise ValueError("At least one audit user is required")
    audit_indices = sorted(set(indices[:audit_count] + [min(indices, key=lambda i: int(dataset[i]["history_lengths"])), max(indices, key=lambda i: int(dataset[i]["history_lengths"]))]))
    audit = audit_model(model, dataset, audit_indices, candidates_t, device)
    write_json(output_dir / "inference_audit.json", audit)
    print(f"Inference audit passed on {audit['users_checked']} users", flush=True)
    del checkpoint_data
    started = time.perf_counter()
    row_count = mask_only_count = 0
    with result_path.open("w", encoding="utf-8", newline="") as destination, (output_dir / "mask_only_results.csv").open("w", encoding="utf-8", newline="") as mask_destination:
        writer = mask_writer = None
        with torch.inference_mode():
            for processed, user_index in enumerate(indices, start=1):
                sample = dataset[user_index]
                length = int(sample["history_lengths"])
                target = int(sample["target_ids"])
                controlled = make_controlled_perturbations(sample, datamodule.all_item_ids, repeats, seed, user_index, tail_count, shifts)
                samples = [sample] + [p.perturbation.sample for p in controlled]
                logits_parts = []
                for start in range(0, len(samples), batch_size):
                    features = features_for(model, samples[start:start + batch_size], device)
                    logits_parts.append(logits_from_features(model, features, candidates_t))
                raw_logits = torch.cat(logits_parts)
                historical_ids = torch.stack([s["historical_ids"] for s in samples]).to(device)
                dynamic_mask = seen_item_mask(candidate_ids, historical_ids)
                fixed_mask = dynamic_mask[:1].expand_as(dynamic_mask)
                outputs = {
                    "dynamic": recommendations_from_logits(raw_logits, candidate_ids, dynamic_mask, [target] * len(samples), top_k),
                    "fixed_original": recommendations_from_logits(raw_logits, candidate_ids, fixed_mask, [target] * len(samples), top_k),
                }
                mask_only_outputs = recommendations_from_logits(raw_logits[:1].expand_as(raw_logits), candidate_ids, dynamic_mask, [target] * len(samples), top_k)
                original = outputs["dynamic"][0]
                legacy_indices = {i for _, i in _unique_position_indices(length)}
                for row_index, controlled_p in enumerate(controlled, start=1):
                    p = controlled_p.perturbation
                    original_time = int(sample["historical_timestamps"][p.position_index])
                    base = {
                        "user_index": user_index, "user_id": int(sample["user_id"]), "history_length": length,
                        "perturbed_history_length": int(p.sample["history_lengths"]), "target_id": target,
                        "target_rating": int(sample["target_ratings"]), "target_timestamp": int(sample["target_timestamps"]),
                        "perturbation_type": p.perturbation_type, "position_label": p.position_label,
                        "position_index": p.position_index, "position_fraction": p.position_index / (length - 1),
                        "distance_from_end": length - 1 - p.position_index, "is_legacy_position": p.position_index in legacy_indices,
                        "original_item_id": p.original_item_id, "perturbed_item_id": p.perturbed_item_id,
                        "original_timestamp": original_time,
                        "perturbed_timestamp": None if p.perturbation_type == "delete" else int(p.sample["historical_timestamps"][p.position_index]),
                        "requested_timestamp_shift_seconds": controlled_p.requested_shift,
                        "actual_timestamp_shift_seconds": controlled_p.actual_shift,
                        "timestamp_changed": controlled_p.timestamp_changed,
                        "changed_time_bucket_count": controlled_p.changed_time_buckets,
                        "repetition": p.repetition, "seed": p.seed, "top_k": top_k,
                        "candidate_mask_changed_count": int((dynamic_mask[row_index] != dynamic_mask[0]).sum()),
                        "original_topk_ids": json.dumps(original.top_ids),
                    }
                    for mode in ("dynamic", "fixed_original"):
                        perturbed = outputs[mode][row_index]
                        row = {**base, "mask_mode": mode, **comparison_metrics(original, perturbed, top_k, target), "perturbed_topk_ids": json.dumps(perturbed.top_ids)}
                        if writer is None:
                            writer = csv.DictWriter(destination, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        row_count += 1
                    if p.perturbation_type != "timestamp_only":
                        mask_row = {**base, "mask_mode": "mask_only", **comparison_metrics(original, mask_only_outputs[row_index], top_k, target), "perturbed_topk_ids": json.dumps(mask_only_outputs[row_index].top_ids)}
                        if mask_writer is None:
                            mask_writer = csv.DictWriter(mask_destination, fieldnames=list(mask_row))
                            mask_writer.writeheader()
                        mask_writer.writerow(mask_row)
                        mask_only_count += 1
                if processed % 25 == 0 or processed == num_users:
                    destination.flush()
                    mask_destination.flush()
                    print(f"Processed {processed}/{num_users}; comparisons={row_count}; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started, rows=row_count, mask_only_rows=mask_only_count,
                    results_sha256=file_sha256(result_path), mask_only_results_sha256=file_sha256(output_dir / "mask_only_results.csv"))
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
