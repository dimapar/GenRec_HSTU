import numpy as np
import pandas as pd

from generative_recommenders_pl.scripts.analyze_controlled_robustness import UserBootstrap, summarize_users, paired_mask_differences


def test_bootstrap_resamples_users_not_replacement_rows():
    rows = pd.DataFrame({"group": ["a"] * 4, "user_index": [0, 1, 2, 3], "metric": [0.0, 0.25, 0.5, 1.0]})
    duplicated = pd.concat([rows] * 5, ignore_index=True)
    bootstrap = UserBootstrap(1000, 42)
    single = summarize_users(rows, ["group"], bootstrap, ("metric",)).iloc[0]
    repeated = summarize_users(duplicated, ["group"], bootstrap, ("metric",)).iloc[0]
    for key in ("n_users", "metric_mean", "metric_ci_low", "metric_ci_high"):
        assert single[key] == repeated[key]
    assert single.n_observations == 4 and repeated.n_observations == 20
    assert single.metric_ci_low <= single.metric_mean <= single.metric_ci_high


def test_paired_contrast_pairs_the_same_perturbations():
    frame = pd.DataFrame([
        {"user_index": user, "perturbation_type": "delete", "position_index": 4,
         "repetition": 0, "requested_timestamp_shift_seconds": 0, "mask_mode": mode,
         "topk_overlap": overlap, "js_divergence": js, "top1_changed": changed}
        for user in range(3)
        for mode, overlap, js, changed in (("dynamic", 0.4, 0.2, True), ("fixed_original", 0.7, 0.1, False))
    ])
    paired = paired_mask_differences(frame)
    np.testing.assert_allclose(paired.topk_overlap, -0.3)
    np.testing.assert_allclose(paired.js_divergence, 0.1)
    np.testing.assert_allclose(paired.top1_changed, 1.0)
    assert len(paired) == 3


def test_bootstrap_is_deterministic_and_reports_single_user_as_unidentified():
    values = np.array([0.1, 0.3, 0.7, 0.9])
    a = UserBootstrap(1000, 42)
    b = UserBootstrap(1000, 42)
    np.testing.assert_array_equal(a.interval(values), b.interval(values))
    a.interval(np.arange(7))
    np.testing.assert_array_equal(a.interval(values), b.interval(values))
    assert np.isnan(a.interval([0.5])).all()
