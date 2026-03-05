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
class ExtractorConfig:
    active_type: Literal["visual_clip"]
    visual_clip_config: VisualClipConfig | None = None # Make it optional and correctly typed

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

        # Validate paths exist
        paths_config = PathsConfig(
            dev_txt=Path(config_data["paths"]["dev_txt"]),
            video_dir=Path(config_data["paths"]["video_dir"]),
            feat_out=Path(config_data["paths"]["feat_out"])
        )
        if not paths_config.dev_txt.exists():
            raise FileNotFoundError(f"dev_txt path does not exist: {paths_config.dev_txt}")
        if not paths_config.video_dir.exists():
            raise FileNotFoundError(f"video_dir path does not exist: {paths_config.video_dir}")
        # feat_out does not need to exist, it will be created

        # Parse extractor config based on active_type
        active_extractor_type = config_data["extractor"]["active_type"]

        visual_clip_config_instance: VisualClipConfig | None = None
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
        else:
            raise ValueError(f"Unsupported active extractor type: {active_extractor_type}")

        extractor_config_obj = ExtractorConfig(
            active_type=active_extractor_type,
            visual_clip_config=visual_clip_config_instance # Pass directly during initialization
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

