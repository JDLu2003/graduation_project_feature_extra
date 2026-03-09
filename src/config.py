from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

@dataclass(frozen=True)
class PathsConfig:
    dev_txt: Path
    video_dir: Path
    feat_out: Path

@dataclass(frozen=True)
class FrameSamplingConfig:
    strategy: Literal["uniform", "middle", "first"]
    num_frames: int
    aggregation: Literal["mean", "max"]

@dataclass(frozen=True)
class VisualClipConfig:
    model_name: str
    device: Literal["cpu", "cuda", "mps"]
    clip_output_dim: int
    target_dim: int
    frame_sampling: FrameSamplingConfig


@dataclass(frozen=True)
class FaceSceneFRConfig:
    """
    配置“人物512 + 环境512”策略。
    """
    device: Literal["cpu", "cuda", "mps"]
    face_checkpoint: Path
    clip_model_name: str
    frame_sampling: FrameSamplingConfig
    person_num_frames: int
    mtcnn_image_size: int
    mtcnn_margin: int
    mtcnn_min_face_size: int
    mtcnn_thresholds: list[float]
    mtcnn_keep_all: bool
    face_batch_size: int
    min_detection_confidence: float
    classification_strategy: Literal["top1", "top1_with_threshold"]
    min_classification_confidence: float
    unknown_person_strategy: Literal["zero", "other_mean"]
    other_label_name: str

@dataclass(frozen=True)
class ExtractorConfig:
    active_type: Literal["visual_clip", "face_scene_fr"]
    visual_clip_config: VisualClipConfig | None = None # Make it optional and correctly typed
    face_scene_fr_config: FaceSceneFRConfig | None = None

@dataclass(frozen=True)
class NonSpeakerConfig:
    strategy: Literal["context_video"]
    fallback: Literal["zero"]

@dataclass(frozen=True)
class PipelineConfig:
    skip_existing: bool
    show_progress: bool

@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    extractor: ExtractorConfig
    non_speaker: NonSpeakerConfig
    pipeline: PipelineConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AppConfig":
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        base_dir = yaml_path.resolve().parent

        # Validate paths exist
        paths_config = PathsConfig(
            dev_txt=(base_dir / config_data["paths"]["dev_txt"]).resolve(),
            video_dir=(base_dir / config_data["paths"]["video_dir"]).resolve(),
            feat_out=(base_dir / config_data["paths"]["feat_out"]).resolve(),
        )
        if not paths_config.dev_txt.exists():
            raise FileNotFoundError(f"dev_txt path does not exist: {paths_config.dev_txt}")
        if not paths_config.video_dir.exists():
            raise FileNotFoundError(f"video_dir path does not exist: {paths_config.video_dir}")
        # feat_out does not need to exist, it will be created

        # Parse extractor config based on active_type
        active_extractor_type = config_data["extractor"]["active_type"]

        visual_clip_config_instance: VisualClipConfig | None = None
        face_scene_fr_config_instance: FaceSceneFRConfig | None = None
        if active_extractor_type == "visual_clip":
            visual_clip_data = config_data.get("visual_clip_config", {})
            frame_sampling_config = FrameSamplingConfig(**visual_clip_data["frame_sampling"])
            visual_clip_config_instance = VisualClipConfig(
                model_name=visual_clip_data["model_name"],
                device=visual_clip_data["device"],
                clip_output_dim=visual_clip_data["clip_output_dim"],
                target_dim=visual_clip_data["target_dim"],
                frame_sampling=frame_sampling_config
            )
        elif active_extractor_type == "face_scene_fr":
            fr_data = config_data.get("face_scene_fr_config", {})
            frame_sampling_config = FrameSamplingConfig(**fr_data["frame_sampling"])
            face_scene_fr_config_instance = FaceSceneFRConfig(
                device=fr_data["device"],
                face_checkpoint=(base_dir / fr_data["face_checkpoint"]).resolve(),
                clip_model_name=fr_data["clip_model_name"],
                frame_sampling=frame_sampling_config,
                person_num_frames=int(fr_data.get("person_num_frames", frame_sampling_config.num_frames)),
                mtcnn_image_size=fr_data["mtcnn_image_size"],
                mtcnn_margin=fr_data["mtcnn_margin"],
                mtcnn_min_face_size=fr_data["mtcnn_min_face_size"],
                mtcnn_thresholds=fr_data["mtcnn_thresholds"],
                mtcnn_keep_all=fr_data["mtcnn_keep_all"],
                face_batch_size=int(fr_data.get("face_batch_size", 64)),
                min_detection_confidence=fr_data["min_detection_confidence"],
                classification_strategy=fr_data.get("classification_strategy", "top1"),
                min_classification_confidence=float(fr_data.get("min_classification_confidence", 0.5)),
                unknown_person_strategy=fr_data.get("unknown_person_strategy", "zero"),
                other_label_name=str(fr_data.get("other_label_name", "other")),
            )
            if not face_scene_fr_config_instance.face_checkpoint.exists():
                raise FileNotFoundError(
                    f"face_checkpoint path does not exist: {face_scene_fr_config_instance.face_checkpoint}"
                )
        else:
            raise ValueError(f"Unsupported active extractor type: {active_extractor_type}")

        extractor_config_obj = ExtractorConfig(
            active_type=active_extractor_type,
            visual_clip_config=visual_clip_config_instance,
            face_scene_fr_config=face_scene_fr_config_instance,
        )

        return cls(
            paths=paths_config,
            extractor=extractor_config_obj,
            non_speaker=NonSpeakerConfig(**config_data["non_speaker"]),
            pipeline=PipelineConfig(**config_data["pipeline"])
        )

# Example usage (for testing purposes, will be removed later or moved to main.py)
if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    config_path = current_dir / "config.yaml"
    try:
        config = AppConfig.from_yaml(config_path)
        print("Configuration loaded successfully:")
        print(config)
    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
