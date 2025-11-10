from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class VideoConfig:
    fps_override: Optional[int]
    tile_grid: Tuple[int, int]


@dataclass
class ProposalConfig:
    engine: str
    input_size: int
    conf_thres: float
    iou_thres: float
    max_det: int


@dataclass
class EmbedderConfig:
    model: str
    input_size: int


@dataclass
class GalleryConfig:
    aug_per_ref: int
    max_refs: int
    memory_max: int
    memory_add_sim_cap: float


@dataclass
class SimilarityConfig:
    init_thresh: float
    adapt_bg_window: int


@dataclass
class AssignerConfig:
    alpha_iou: float
    beta_app: float
    iou_gate: float
    sim_gate: float
    cost_gate: float
    new_track_sim: float
    max_age: int


@dataclass
class TrackerConfig:
    engine: str
    gap_trigger: int
    search_scale: float


@dataclass
class PostConfig:
    smooth_window: int
    track_only_max: int


@dataclass
class RuntimeConfig:
    device: str
    batch_embed: int


@dataclass
class PipelineConfig:
    video: VideoConfig
    proposals: ProposalConfig
    embedder: EmbedderConfig
    gallery: GalleryConfig
    similarity: SimilarityConfig
    assigner: AssignerConfig
    tracker: TrackerConfig
    post: PostConfig
    runtime: RuntimeConfig

    def copy_with(self, **kwargs: Any) -> "PipelineConfig":
        data = {**self.__dict__, **kwargs}
        return PipelineConfig(**data)


def _tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected length-2 tuple list, got {value!r}")


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw: Dict[str, Dict[str, Any]] = yaml.safe_load(fp)

    return PipelineConfig(
        video=VideoConfig(
            fps_override=raw["video"].get("fps_override"),
            tile_grid=_tuple(raw["video"].get("tile_grid", (1, 1))),
        ),
        proposals=ProposalConfig(**raw["proposals"]),
        embedder=EmbedderConfig(**raw["embedder"]),
        gallery=GalleryConfig(**raw["gallery"]),
        similarity=SimilarityConfig(**raw["similarity"]),
        assigner=AssignerConfig(**raw["assigner"]),
        tracker=TrackerConfig(**raw["tracker"]),
        post=PostConfig(**raw["post"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )


DEFAULT_CONFIG_PATH = Path(__file__).with_suffix("").parent / "cfg" / "default.yaml"
