"""End-to-end deletion and embedding-interface checks on a loaded model."""

import torch

from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import seq_features_from_row
from generative_recommenders_pl.models.utils.inference import logits_from_features, logits_from_embeddings
from generative_recommenders_pl.scripts.run_robustness import _batch_samples, _delete_item, _clone_sample


def features_for(model, samples, device):
    features, _, _ = seq_features_from_row(
        _batch_samples(samples), device, model.gr_output_length + 1
    )
    return features._replace(past_embeddings=model.embeddings.get_item_embeddings(features.past_ids))


def audit_model(model, dataset, indices, candidate_embeddings_t, device):
    maximum_batch_error = 0.0
    maximum_embedding_error = 0.0
    checked = 0
    model.eval()
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            length = int(sample["history_lengths"])
            if length <= 1:
                continue
            deleted = _delete_item(sample, length - 1)
            rebuilt = dict(sample)
            # Independent construction from the real prefix, not from _delete_item.
            for key in ("historical_ids", "historical_ratings", "historical_timestamps"):
                values = sample[key][:length - 1].tolist()
                rebuilt[key] = torch.tensor(values + [0] * (sample[key].numel() - len(values)), dtype=sample[key].dtype)
                assert torch.equal(rebuilt[key], deleted[key]), key
                assert not deleted[key][length - 1:].any(), key
            rebuilt["history_lengths"] = length - 1
            assert deleted["history_lengths"] == length - 1
            features = features_for(model, [sample, deleted, rebuilt], device)
            for row, item in enumerate((sample, deleted, rebuilt)):
                real_length = int(item["history_lengths"])
                assert int((features.past_ids[row] != 0).sum()) == real_length
                assert int(features.past_payloads["timestamps"][row, real_length]) == int(item["target_timestamps"])
                assert torch.equal(features.past_payloads["timestamps"][row, :real_length].cpu(), item["historical_timestamps"][:real_length])
            assert torch.equal(model.sequence_encoder._attn_mask, torch.triu(torch.ones_like(model.sequence_encoder._attn_mask), diagonal=1))
            encoded, _ = model(features)
            current = ops.get_current_embeddings(features.past_lengths, encoded)
            torch.testing.assert_close(current[1], encoded[1, length - 2], rtol=0, atol=0)
            torch.testing.assert_close(encoded[1, :length - 1], encoded[2, :length - 1], rtol=0, atol=0)
            batched = current @ candidate_embeddings_t
            single_features = features_for(model, [deleted], device)
            single = logits_from_features(model, single_features, candidate_embeddings_t)[0]
            error = float((batched[1] - single).abs().max())
            maximum_batch_error = max(maximum_batch_error, error)
            torch.testing.assert_close(batched[1], single, rtol=1e-4, atol=2e-6)
            embedded = logits_from_embeddings(model, features.past_embeddings, features.past_lengths, features.past_payloads["timestamps"], candidate_embeddings_t)
            maximum_embedding_error = max(maximum_embedding_error, float((embedded - batched).abs().max()))
            torch.testing.assert_close(embedded, batched, rtol=0, atol=0)
            # Padding timestamps AFTER the prediction-time anchor cannot affect
            # any active query/key, although the anchor itself can affect it.
            dirty_times = single_features.past_payloads["timestamps"].clone()
            dirty_times[:, length:] = 123456789
            dirty = single_features._replace(past_payloads={**single_features.past_payloads, "timestamps": dirty_times})
            torch.testing.assert_close(logits_from_features(model, dirty, candidate_embeddings_t)[0], single, rtol=0, atol=0)
            checked += 1

        sample = dataset[indices[0]]
        empty = _clone_sample(sample)
        empty["history_lengths"] = 0
        for key in ("historical_ids", "historical_ratings", "historical_timestamps"):
            empty[key].zero_()
        try:
            logits_from_features(model, features_for(model, [empty], device), candidate_embeddings_t)
        except ValueError:
            pass
        else:
            raise AssertionError("An empty history was incorrectly accepted")

    # Check graph construction only; do not compute a Jacobian or any gradients.
    with torch.no_grad():
        features = features_for(model, [dataset[indices[0]]], device)
    embeddings = features.past_embeddings.detach().clone().requires_grad_(True)
    differentiable = logits_from_embeddings(model, embeddings, features.past_lengths, features.past_payloads["timestamps"], candidate_embeddings_t.detach())
    assert differentiable.requires_grad and differentiable.grad_fn is not None
    del differentiable
    return {
        "status": "passed", "users_checked": checked,
        "batch_vs_single_max_abs_logit_error": maximum_batch_error,
        "embedding_vs_id_max_abs_logit_error": maximum_embedding_error,
        "independent_reconstruction": "passed", "updated_lengths_and_last_real_state": "passed",
        "causal_attention_mask": "passed", "prediction_timestamp_anchor": "passed",
        "padding_timestamp_invariance": "passed", "empty_history_rejected": "passed",
        "embedding_autograd_graph": "passed; no derivatives computed",
    }
