"""Differentiable, unmasked scores for the positional HSTU configuration.

Embeddings are supplied BEFORE positional preprocessing. Candidate embeddings
are an explicit input, so changing the history never changes the candidate pool.
No inference_mode/no_grad decorator is used; callers choose whether to keep the
autograd graph. eval() is still required for deterministic predictions.
"""

import torch

from generative_recommenders_pl.models.preprocessors.learnable_positional_embedding import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders_pl.models.utils.features import SequentialFeatures
from generative_recommenders_pl.models.utils.ops import get_current_embeddings


def logits_from_features(model, features, candidate_embeddings_t):
    if (features.past_lengths <= 0).any():
        raise ValueError("Prediction requires a nonempty history")
    if features.past_embeddings is None:
        features = features._replace(
            past_embeddings=model.embeddings.get_item_embeddings(features.past_ids)
        )
    encoded, _ = model(features)
    current = get_current_embeddings(features.past_lengths, encoded)
    return current @ candidate_embeddings_t


def logits_from_embeddings(
    model,
    embeddings: torch.Tensor,
    lengths: torch.Tensor,
    timestamps: torch.Tensor,
    candidate_embeddings_t: torch.Tensor,
) -> torch.Tensor:
    """Return [B,C] raw logits from [B,N,D] item embeddings, without movie IDs.

    N must match the configured encoder input size, including output padding.
    timestamps is [B,N] and includes the prediction timestamp at position L,
    matching seq_features_from_row. lengths is [B]; D=50 for the baseline.
    This interface supports only the ordinary positional preprocessor: auxiliary
    ID-dependent feature preprocessors need their own explicit feature inputs.
    """
    if type(model.preprocessor) is not LearnablePositionalEmbeddingInputFeaturesPreprocessor:
        raise TypeError("Embedding-only inference supports the positional preprocessor")
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [B,N,D]")
    batch, width, dim = embeddings.shape
    if lengths.shape != (batch,) or timestamps.shape != (batch, width):
        raise ValueError("lengths/timestamps do not match embeddings")
    if lengths.dtype != torch.long or timestamps.dtype != torch.long:
        raise ValueError("lengths and timestamps must be int64")
    if (lengths <= 0).any() or (lengths > width).any():
        raise ValueError("Prediction requires lengths in [1,N]")
    if width != model.sequence_encoder._attn_mask.size(0):
        raise ValueError("N must match the configured encoder input size")
    if dim != model.item_embedding_dim or candidate_embeddings_t.ndim != 2:
        raise ValueError("Invalid embedding dimensions")
    if candidate_embeddings_t.size(0) != dim:
        raise ValueError("Candidate embeddings must have shape [D,C]")
    if not (embeddings.device == lengths.device == timestamps.device == candidate_embeddings_t.device):
        raise ValueError("All inputs must be on the same device")
    # Only zero/nonzero validity is consulted by this preprocessor. These are
    # validity indicators, NOT movie IDs and never used for an embedding lookup.
    valid_ids = (torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]).long()
    features = SequentialFeatures(lengths, valid_ids, embeddings, {"timestamps": timestamps})
    return logits_from_features(model, features, candidate_embeddings_t)


def seen_item_mask(candidate_ids: torch.Tensor, historical_ids: torch.Tensor) -> torch.Tensor:
    """Return [B,C] excluded candidates; padding ID zero is never a movie."""
    return ((candidate_ids[None, :, None] == historical_ids[:, None, :])
            & (historical_ids[:, None, :] != 0)).any(dim=2)
