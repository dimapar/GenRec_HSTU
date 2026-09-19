from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from generative_recommenders_pl.models.utils.features import seq_features_from_row
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")


def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def _load_movie_titles(movies_csv: str) -> dict[int, str]:
    movies_path = Path(movies_csv)
    if not movies_path.exists():
        log.warning(f"Movie metadata not found at {movies_path}. Printing ids only.")
        return {}

    movies = pd.read_csv(movies_path)
    return {
        int(row.movie_id): str(row.title)
        for row in movies.itertuples(index=False)
    }


def _movie_label(movie_id: int, title_by_id: dict[int, str]) -> str:
    title = title_by_id.get(movie_id)
    return f"{movie_id}: {title}" if title is not None else str(movie_id)


def _resolve_sample_index(dataset, user_id: int | None, user_index: int) -> int:
    if user_id is None:
        if user_index < 0 or user_index >= len(dataset):
            raise IndexError(
                f"user_index={user_index} is outside dataset range [0, {len(dataset) - 1}]"
            )
        return user_index

    matches = dataset.ratings_frame.index[dataset.ratings_frame["user_id"] == user_id]
    if len(matches) == 0:
        raise ValueError(f"user_id={user_id} was not found in the test dataset")
    return int(matches[0])


def _make_single_batch(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    return default_collate([sample])


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain a state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=False)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt_path is None or cfg.ckpt_path == "":
        raise ValueError("Please provide ckpt_path=... for inspection")

    top_k = int(_cfg_get(cfg, "top_k", 10))
    user_id = cfg.get("user_id", None)
    user_id = None if user_id is None else int(user_id)
    user_index = int(_cfg_get(cfg, "user_index", 0))
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
    model.candidate_index.update_embeddings(
        model.negatives_sampler.normalize_embeddings(
            model.embeddings.get_item_embeddings(model.candidate_index.ids)
        )
    )

    dataset = datamodule.test_dataset
    sample_index = _resolve_sample_index(dataset, user_id=user_id, user_index=user_index)
    sample = dataset[sample_index]
    batch = _make_single_batch(sample)

    seq_features, target_ids, target_ratings = seq_features_from_row(
        batch,
        device=device,
        max_output_length=model.gr_output_length + 1,
    )
    input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
    seq_features = seq_features._replace(past_embeddings=input_embeddings)

    with torch.inference_mode():
        top_k_ids, top_k_scores = model.retrieve(seq_features, filter_past_ids=True)

    title_by_id = _load_movie_titles(datamodule.data_preprocessor.processed_item_csv())

    user_id_value = int(sample["user_id"])
    history_length = int(sample["history_lengths"])
    history_ids = sample["historical_ids"][:history_length].tolist()
    history_ratings = sample["historical_ratings"][:history_length].tolist()
    target_id = int(target_ids.item())
    target_rating = int(target_ratings.item())
    top_ids = [int(x) for x in top_k_ids[0].detach().cpu().tolist()]
    top_scores = [float(x) for x in top_k_scores[0].detach().cpu().tolist()]
    target_rank = next(
        (rank for rank, item_id in enumerate(top_ids, start=1) if item_id == target_id),
        None,
    )

    print("\n=== User sample ===")
    print(f"dataset_index: {sample_index}")
    print(f"user_id: {user_id_value}")
    print(f"device: {device}")
    print(f"checkpoint: {cfg.ckpt_path}")

    print("\n=== Ordered history ===")
    print(f"history_length: {history_length}")
    for position, (movie_id, rating) in enumerate(
        zip(history_ids, history_ratings),
        start=1,
    ):
        print(f"{position:03d}. {_movie_label(int(movie_id), title_by_id)} | rating={int(rating)}")

    print("\n=== Held-out target ===")
    print(f"{_movie_label(target_id, title_by_id)} | rating={target_rating}")
    if target_rank is not None:
        print(f"target_rank_in_top_{len(top_ids)}: {target_rank}")
    else:
        print(f"target_rank_in_top_{len(top_ids)}: not found")

    print(f"\n=== Top-{top_k} recommendations ===")
    for rank, (movie_id, score) in enumerate(
        zip(top_ids[:top_k], top_scores[:top_k]),
        start=1,
    ):
        print(f"{rank:02d}. {_movie_label(movie_id, title_by_id)} | score={score:.6f}")


if __name__ == "__main__":
    main()
