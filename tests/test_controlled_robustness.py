from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from generative_recommenders_pl.models.utils.inference import logits_from_embeddings, logits_from_features, seen_item_mask
from generative_recommenders_pl.scripts.audit_robustness import audit_model, features_for
from generative_recommenders_pl.scripts.controlled_robustness import (
    changed_time_bucket_count, extended_positions, make_controlled_perturbations,
    recommendations_from_logits, shift_timestamp,
)
from generative_recommenders_pl.scripts.run_robustness import _delete_item, _make_perturbations, _score_samples


def sample(length=4, width=8):
    return {
        "user_id": 1, "history_lengths": length,
        "historical_ids": torch.tensor(list(range(1, length + 1)) + [0] * (width - length)),
        "historical_ratings": torch.tensor([4] * length + [0] * (width - length)),
        "historical_timestamps": torch.tensor([10 ** (i + 1) for i in range(length)] + [0] * (width - length)),
        "target_ids": 29, "target_ratings": 5, "target_timestamps": 10 ** (length + 1),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(42)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="eval", overrides=["data.max_sequence_length=8", "model.gr_output_length=2", "data.data_preprocessor.expected_max_item_id=30"])
    dm = SimpleNamespace(all_item_ids=list(range(1, 31)), max_item_id=30, max_sequence_length=8)
    return hydra.utils.instantiate(cfg.model, datamodule=dm, _recursive_=False).eval()


def test_deletion_updates_all_fields_without_mutating_input():
    original = sample()
    deleted = _delete_item(original, 3)
    assert original["history_lengths"] == 4
    assert original["historical_ids"].tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert deleted["history_lengths"] == 3
    assert deleted["historical_ids"].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert deleted["historical_timestamps"].tolist() == [10, 100, 1000, 0, 0, 0, 0, 0]
    assert deleted["historical_ratings"].tolist() == [4, 4, 4, 0, 0, 0, 0, 0]
    with pytest.raises(IndexError):
        _delete_item(original, 4)


def test_extended_positions_preserve_legacy_replacements():
    original = sample(4)
    legacy = _make_perturbations(original, list(range(1, 31)), 5, 42, 17)
    extended = make_controlled_perturbations(original, list(range(1, 31)), 5, 42, 17)
    for old, new in zip(legacy, extended):
        assert (old.position_label, old.position_index, old.seed, old.perturbed_item_id) == (
            new.perturbation.position_label, new.perturbation.position_index, new.perturbation.seed, new.perturbation.perturbed_item_id)
        for key in original:
            if torch.is_tensor(original[key]):
                assert torch.equal(old.sample[key], new.perturbation.sample[key])
    assert len(extended_positions(200)) == 14
    assert len({index for _, index in extended_positions(19)}) == len(extended_positions(19))


def test_timestamp_shift_is_clipped_and_keeps_movie_ids():
    original = sample()
    changed, actual = shift_timestamp(original, 2, 604800)
    assert int(changed["historical_timestamps"][2]) == 10000
    assert actual == 9000
    assert torch.equal(changed["historical_ids"], original["historical_ids"])
    assert changed["history_lengths"] == original["history_lengths"]
    changed, _ = shift_timestamp(original, 3, 604800)
    assert int(changed["historical_timestamps"][3]) == original["target_timestamps"]
    original["historical_timestamps"][:4] = 100
    changed, actual = shift_timestamp(original, 1, 3600)
    assert actual == 0


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("shift", [-604800, -3600, 3600, 604800])
def test_bucket_diagnostic_matches_full_causal_matrix(index, shift):
    original = sample()
    changed, _ = shift_timestamp(original, index, shift)
    def full(s):
        times = torch.cat((s["historical_timestamps"][:4], torch.tensor([s["target_timestamps"]])))
        differences = times[1:, None] - times[None, :-1]
        return (torch.log(differences.abs().clamp(min=1)) / 0.301).long().clamp(0, 128)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    expected = int(((full(original) != full(changed)) & mask).sum())
    assert changed_time_bucket_count(original, index, int(changed["historical_timestamps"][index])) == expected


def test_masks_can_change_predictions_with_identical_logits():
    ids = torch.arange(1, 16)
    histories = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 0]])
    masks = seen_item_mask(ids, histories)
    raw = torch.zeros(2, 15)
    raw[:, 2] = 10
    raw[:, 3] = 9
    dynamic = recommendations_from_logits(raw, ids, masks, [15, 15], 10)
    fixed = recommendations_from_logits(raw, ids, masks[:1].expand_as(masks), [15, 15], 10)
    assert dynamic[0].top_ids[0] == 4 and dynamic[1].top_ids[0] == 3
    assert fixed[0].top_ids == fixed[1].top_ids
    assert torch.equal(fixed[0].probabilities, fixed[1].probabilities)


def test_real_hstu_deletion_audit_and_embedding_interface(model):
    dataset = [sample(length) for length in (2, 4, 8)]
    with torch.no_grad():
        candidates = model.negatives_sampler.normalize_embeddings(model.embeddings.get_item_embeddings(torch.arange(1, 31))).t().contiguous()
    audit = audit_model(model, dataset, [0, 1, 2], candidates, torch.device("cpu"))
    assert audit["status"] == "passed" and audit["users_checked"] == 3
    features = features_for(model, [sample()], torch.device("cpu"))
    embeddings = features.past_embeddings.detach().requires_grad_(True)
    # With all model parameters frozen, a graph can only come from the supplied
    # leaf embeddings. This catches an accidental detach of the input without
    # computing derivatives or a Jacobian.
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for parameter, _ in parameter_flags:
            parameter.requires_grad_(False)
        output = logits_from_embeddings(model, embeddings, features.past_lengths, features.past_payloads["timestamps"], candidates.detach())
        assert output.requires_grad and output.grad_fn is not None
    finally:
        for parameter, flag in parameter_flags:
            parameter.requires_grad_(flag)
    with pytest.raises(ValueError, match="nonempty"):
        logits_from_features(model, features._replace(past_lengths=torch.zeros_like(features.past_lengths)), candidates)
    with pytest.raises(ValueError, match="nonempty"):
        _score_samples(model, [_delete_item(sample(1), 0)], torch.arange(1, 31), candidates, [29], 10, torch.device("cpu"))


def test_scoring_with_dynamic_mask_matches_legacy_path(model):
    samples = [sample(), _delete_item(sample(), 3)]
    ids = torch.arange(1, 31)
    with torch.no_grad():
        candidates = model.negatives_sampler.normalize_embeddings(model.embeddings.get_item_embeddings(ids)).t().contiguous()
        features = features_for(model, samples, torch.device("cpu"))
        raw = logits_from_features(model, features, candidates)
        new = recommendations_from_logits(raw, ids, seen_item_mask(ids, features.past_ids), [29, 29], 10)
        old = _score_samples(model, samples, ids, candidates, [29, 29], 10, torch.device("cpu"))
    for before, after in zip(old, new):
        assert before.top_ids == after.top_ids
        assert before.target_rank == after.target_rank
        torch.testing.assert_close(before.probabilities, after.probabilities, rtol=0, atol=0)
