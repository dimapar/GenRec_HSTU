"""User-level bootstrap, paired contrasts and regression checks for stage two.

Confidence intervals are pointwise percentile intervals, conditional on the
fixed model, cohort, and sampled replacements. Repeated observations from one
user are averaged BEFORE resampling users. The same bootstrap counts are reused
for groups with the same user count; the replacement RNG is never touched.
"""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("topk_overlap", "js_divergence", "top1_changed")
GROUP_BASE = ["perturbation_type", "mask_mode", "requested_timestamp_shift_seconds"]


class UserBootstrap:
    def __init__(self, repeats=5000, seed=42):
        if repeats < 100:
            raise ValueError("Use at least 100 bootstrap replicates")
        self.repeats, self.seed = repeats, seed
        self._weights = OrderedDict()

    def interval(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        users = len(values)
        if users < 2:
            return np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
        if users not in self._weights:
            # Independent from movie replacement generators; reproducible for
            # each cohort size and independent of summary traversal order.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, users]))
            self._weights[users] = rng.multinomial(users, np.full(users, 1 / users), size=self.repeats).astype(np.float64) / users
            if len(self._weights) > 8:
                self._weights.popitem(last=False)
        self._weights.move_to_end(users)
        draws = self._weights[users] @ values
        return tuple(np.quantile(draws, [0.025, 0.975], axis=0))


def summarize_users(frame, group_columns, bootstrap, metrics=METRICS):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group.groupby("user_index", sort=True)[list(metrics)].mean()
        low, high = bootstrap.interval(values.to_numpy())
        row = dict(zip(group_columns, keys))
        row.update(n_users=len(values), n_observations=len(group))
        for column, metric in enumerate(metrics):
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_ci_low"] = float(low[column])
            row[f"{metric}_ci_high"] = float(high[column])
        rows.append(row)
    return pd.DataFrame(rows, columns=group_columns + ["n_users", "n_observations"] + [f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "ci_low", "ci_high")])


def paired_mask_differences(frame):
    keys = ["user_index", "perturbation_type", "position_index", "repetition", "requested_timestamp_shift_seconds"]
    dynamic = frame.loc[frame.mask_mode == "dynamic"].copy()
    fixed = frame.loc[frame.mask_mode == "fixed_original", keys + list(METRICS)].copy()
    merged = dynamic.merge(fixed, on=keys, validate="one_to_one", suffixes=("_dynamic", "_fixed"))
    for metric in METRICS:
        merged[metric] = merged[f"{metric}_dynamic"].astype(float) - merged[f"{metric}_fixed"].astype(float)
    merged["mask_mode"] = "dynamic_minus_fixed"
    return merged


def tail_contrasts(frame):
    means = frame.loc[frame.distance_from_end < 10].groupby(
        GROUP_BASE + ["user_index", "distance_from_end"], sort=True
    )[list(METRICS)].mean().reset_index()
    last = means.loc[means.distance_from_end == 0]
    pieces = []
    for distance in (1, 9):
        other = means.loc[means.distance_from_end == distance]
        joined = last.merge(other, on=GROUP_BASE + ["user_index"], validate="one_to_one", suffixes=("_last", "_other"))
        for metric in METRICS:
            joined[metric] = joined[f"{metric}_last"] - joined[f"{metric}_other"]
        joined["contrast"] = f"last_minus_distance_{distance}"
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True)


def regression_validation(frame, directory, reference_path):
    validation = {}
    keys = ["user_index", "perturbation_type", "position_label", "repetition"]
    legacy = frame.loc[(frame.mask_mode == "dynamic") & frame.is_legacy_position & (frame.perturbation_type != "timestamp_only")]
    reference = pd.read_csv(reference_path)
    original_path = directory / "reference_reproduction.csv"
    if original_path.exists():
        reproduction = pd.read_csv(original_path)
        validation["independent_legacy_reproduction_all_columns_equal"] = bool(reference.equals(reproduction))
    if set(legacy.user_index) != set(reference.user_index):
        validation["legacy_comparison"] = "not performed: different user cohort"
    else:
        joined = reference.merge(legacy, on=keys, validate="one_to_one", suffixes=("_old", "_new"))
        if len(joined) != len(reference) or len(joined) != len(legacy):
            raise AssertionError("Incomplete legacy-position comparison")
        validation["legacy_comparison_rows"] = len(joined)
        validation["legacy_max_abs_metric_delta"] = {
            metric: float((joined[f"{metric}_old"].astype(float) - joined[f"{metric}_new"].astype(float)).abs().max()) for metric in METRICS
        }
        for field in ("seed", "perturbed_item_id", "original_topk_ids", "perturbed_topk_ids"):
            validation[f"legacy_{field}_mismatches"] = int((joined[f"{field}_old"].fillna("NA") != joined[f"{field}_new"].fillna("NA")).sum())
        if validation["legacy_max_abs_metric_delta"]["topk_overlap"] or validation["legacy_max_abs_metric_delta"]["top1_changed"]:
            raise AssertionError("Legacy recommendations changed")
        validation["legacy_js_tolerance"] = {"rtol": 1e-6, "atol": 1e-11, "reason": "float32 CUDA rounding with different inference batch shapes; rankings must agree exactly"}
        if not np.allclose(joined.js_divergence_old, joined.js_divergence_new, rtol=1e-6, atol=1e-11):
            raise AssertionError("Legacy JS regression")
        if validation["legacy_seed_mismatches"] or validation["legacy_perturbed_item_id_mismatches"]:
            raise AssertionError("Replacement protocol changed")
        if validation["legacy_original_topk_ids_mismatches"] or validation["legacy_perturbed_topk_ids_mismatches"]:
            raise AssertionError("Legacy Top-10 ordering changed")
    time_only = frame.loc[frame.perturbation_type == "timestamp_only"]
    paired = paired_mask_differences(time_only)
    validation["time_only_mask_modes_max_abs_metric_difference"] = {m: float(paired[m].abs().max()) for m in METRICS}
    if any(validation["time_only_mask_modes_max_abs_metric_difference"].values()):
        raise AssertionError("Timestamp-only mask modes disagree")
    unchanged_buckets = time_only.loc[time_only.changed_time_bucket_count == 0]
    validation["unchanged_bucket_rows"] = len(unchanged_buckets)
    validation["unchanged_bucket_max_js"] = float(unchanged_buckets.js_divergence.abs().max())
    validation["unchanged_bucket_recommendation_changes"] = int(((unchanged_buckets.topk_overlap != 1) | unchanged_buckets.top1_changed).sum())
    if validation["unchanged_bucket_recommendation_changes"] or validation["unchanged_bucket_max_js"] > 1e-12:
        raise AssertionError("Predictions changed despite identical time buckets and items")
    validation["status"] = "passed"
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/robustness/stage2"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/robustness/stage1/ml1m_hstu_robustness.csv"))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    directory = args.input_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Only completed experiments can be analyzed")
    frame = pd.read_csv(directory / "controlled_results.csv")
    mask_only = pd.read_csv(directory / "mask_only_results.csv")
    if len(frame) != manifest["rows"] or len(mask_only) != manifest["mask_only_rows"]:
        raise AssertionError("Result counts do not match the completed manifest")
    bootstrap = UserBootstrap(args.bootstrap_repeats, args.bootstrap_seed)
    summaries = {
        "summary_legacy_positions.csv": summarize_users(frame.loc[frame.is_legacy_position], GROUP_BASE + ["position_label"], bootstrap),
        "summary_tail_positions.csv": summarize_users(frame.loc[frame.distance_from_end < 10], GROUP_BASE + ["distance_from_end"], bootstrap),
        "summary_mask_only_legacy.csv": summarize_users(mask_only.loc[mask_only.is_legacy_position], GROUP_BASE + ["position_label"], bootstrap),
        "summary_mask_only_tail.csv": summarize_users(mask_only.loc[mask_only.distance_from_end < 10], GROUP_BASE + ["distance_from_end"], bootstrap),
    }
    paired = paired_mask_differences(frame)
    summaries["paired_mask_differences_legacy.csv"] = summarize_users(paired.loc[paired.is_legacy_position], GROUP_BASE + ["position_label"], bootstrap)
    summaries["paired_mask_differences_tail.csv"] = summarize_users(paired.loc[paired.distance_from_end < 10], GROUP_BASE + ["distance_from_end"], bootstrap)
    contrasts = tail_contrasts(frame)
    summaries["tail_contrasts.csv"] = summarize_users(contrasts, GROUP_BASE + ["contrast"], bootstrap)
    time_only = frame.loc[(frame.perturbation_type == "timestamp_only") & (frame.mask_mode == "fixed_original")].copy()
    time_only["timestamp_clipped"] = time_only.actual_timestamp_shift_seconds != time_only.requested_timestamp_shift_seconds
    time_only["time_bucket_changed"] = time_only.changed_time_bucket_count > 0
    time_only["absolute_actual_shift_seconds"] = time_only.actual_timestamp_shift_seconds.abs()
    diagnostics = ("timestamp_changed", "timestamp_clipped", "time_bucket_changed", "absolute_actual_shift_seconds", "changed_time_bucket_count")
    summaries["timestamp_diagnostics_legacy.csv"] = summarize_users(time_only.loc[time_only.is_legacy_position], ["requested_timestamp_shift_seconds", "position_label"], bootstrap, diagnostics)
    summaries["timestamp_diagnostics_tail.csv"] = summarize_users(time_only.loc[time_only.distance_from_end < 10], ["requested_timestamp_shift_seconds", "distance_from_end"], bootstrap, diagnostics)
    changed = time_only.loc[time_only.time_bucket_changed]
    summaries["timestamp_changed_buckets_legacy.csv"] = summarize_users(changed.loc[changed.is_legacy_position], GROUP_BASE + ["position_label"], bootstrap)
    summaries["timestamp_changed_buckets_tail.csv"] = summarize_users(changed.loc[changed.distance_from_end < 10], GROUP_BASE + ["distance_from_end"], bootstrap)
    for name, summary in summaries.items():
        summary.to_csv(directory / name, index=False)
    validation = regression_validation(frame, directory, args.reference)
    (directory / "analysis_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.bootstrap_seed,
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy_version": np.__version__, "pandas_version": pd.__version__,
                "method": "pointwise 95% percentile bootstrap over users after within-user averaging",
                "paired_difference_direction": "dynamic minus fixed_original",
                "timestamp_primary_means_include_clipped_and_noop_cases": True,
                "timestamp_changed_buckets_tables": "conditional diagnostic; n_users varies",
                "interpretation": "conditional on fixed checkpoint, selected users and sampled replacements; first 500 users are not a random population sample"}
    (directory / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analyzed {len(frame)} comparisons from {frame.user_index.nunique()} users; wrote {len(summaries)} tables")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
