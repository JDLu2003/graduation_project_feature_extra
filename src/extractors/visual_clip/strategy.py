from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from src.extractors.base import FeatureExtractor
from src.extractors.visual_clip.clip_encoder import CLIPEncoder
from src.extractors.visual_clip.context_index import build_context_index
from src.config import VisualClipConfig, NonSpeakerConfig
from src.parser import DialogueRecord, UtteranceRecord

class VisualCLIPStrategy(FeatureExtractor):
    """
    A concrete implementation of FeatureExtractor using CLIP for visual features.
    This strategy handles both speaker and non-speaker feature extraction
    by integrating a CLIPEncoder and a context index for non-speakers.
    """
    def __init__(self, config: VisualClipConfig, non_speaker_config: NonSpeakerConfig):
        self.config = config
        self.non_speaker_config = non_speaker_config
        self.clip_encoder = CLIPEncoder(config)
        self.runtime_device = self.clip_encoder.runtime_device
        self._output_dim = self.clip_encoder.output_dim
        self.context_index: Dict[Tuple[int, str], List[Path]] = {} # Will be built in prepare()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def prepare(self, dialogue_records: List[DialogueRecord], video_base_dir: Path) -> None:
        """
        Prepares the strategy by building the context index for non-speaker lookup.
        """
        print(f"[{self.__class__.__name__}] Building context index for non-speakers...")
        self.context_index = build_context_index(dialogue_records, video_base_dir)
        print(f"[{self.__class__.__name__}] Context index built with {len(self.context_index)} entries.")

    def extract_speaker(self, video_path: Path) -> torch.Tensor:
        """
        Extracts features for a speaker by encoding their video.
        """
        return self.clip_encoder.encode_video_frames(video_path)

    def extract_non_speaker(self, person_name: str, dialogue_id: int) -> torch.Tensor:
        """
        Extracts features for a non-speaker using the context index.
        If videos are found, their features are averaged. Otherwise, a zero vector is returned.
        """
        key = (dialogue_id, person_name)
        if key in self.context_index and self.context_index[key]:
            video_paths = self.context_index[key]
            print(f"[VisualCLIPStrategy] Found {len(video_paths)} context video(s) for '{person_name}' in dialogue {dialogue_id}.")
            # Extract features for all videos where this person is a speaker in this dialogue
            all_features: List[torch.Tensor] = []
            for video_path in video_paths:
                if video_path.exists():
                    all_features.append(self.clip_encoder.encode_video_frames(video_path))
                else:
                    print(f"[VisualCLIPStrategy] Warning: Context video not found: '{video_path}', skipping.")

            if all_features:
                # Average the features if multiple videos are found
                averaged_features = torch.mean(torch.stack(all_features), dim=0, keepdim=False)
                # Re-normalize the averaged features
                return F.normalize(averaged_features, p=2, dim=-1).to(torch.float64)
            else:
                # Fallback to zero vector if videos in index don't exist
                print(f"[{self.__class__.__name__}] Warning: No valid video paths found for non-speaker '{person_name}' in dialogue {dialogue_id}, returning zero vector.")
                return torch.zeros(1, self.output_dim, dtype=torch.float64)
        else:
            # If no context videos are available, return a zero vector as fallback
            print(f"[{self.__class__.__name__}] No context videos found for non-speaker '{person_name}' in dialogue {dialogue_id}, returning zero vector.")
            return torch.zeros(1, self.output_dim, dtype=torch.float64)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Adjust sys.path for direct execution to find project root modules
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent # Assuming src is directly under project_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # from src.config import AppConfig, VisualClipConfig, NonSpeakerConfig # Already imported above
    from src.parser import parse_dev_txt, DialogueRecord, UtteranceRecord
    from src.config import AppConfig, VisualClipConfig, NonSpeakerConfig

    print("--- Starting VisualCLIPStrategy Self-Test ---")

    # Load configuration
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        print(f"Configuration loaded from {config_path}")

        # Instantiate VisualCLIPStrategy
        clip_config: VisualClipConfig = app_config.extractor.visual_clip_config
        non_speaker_config: NonSpeakerConfig = app_config.non_speaker

        if not clip_config or not non_speaker_config:
            raise AssertionError("VisualClipConfig or NonSpeakerConfig is missing in config.yaml")

        strategy = VisualCLIPStrategy(clip_config, non_speaker_config)
        print(f"VisualCLIPStrategy initialized with output dim: {strategy.output_dim}")

        # Prepare the strategy by building the context index
        dev_txt_path = app_config.paths.dev_txt
        video_base_dir = app_config.paths.video_dir
        if not dev_txt_path.exists():
            raise FileNotFoundError(f"dev.txt not found at {dev_txt_path}")
        if not video_base_dir.exists():
            raise FileNotFoundError(f"Video base directory not found at {video_base_dir}")

        dialogue_records: List[DialogueRecord] = parse_dev_txt(dev_txt_path)
        if not dialogue_records:
            raise ValueError("No dialogue records parsed from dev.txt. Cannot proceed with testing.")

        strategy.prepare(dialogue_records, video_base_dir)

        # Test extract_speaker
        print("\n--- Testing Speaker Feature Extraction ---")
        # Find a valid video for a speaker
        test_utterance: UtteranceRecord = dialogue_records[0].utterances[0]
        dialogue_folder_name = f"C_{test_utterance.dialogue_id}"
        video_file_name = f"C_{test_utterance.dialogue_id}_U_{test_utterance.utterance_idx}.mp4"
        speaker_video_path = video_base_dir / dialogue_folder_name / video_file_name

        if not speaker_video_path.exists():
            print(f"Warning: Speaker test video not found at {speaker_video_path}. Skipping speaker test.")
        else:
            speaker_feat = strategy.extract_speaker(speaker_video_path)
            assert isinstance(speaker_feat, torch.Tensor), "Speaker feature is not a Tensor."
            assert speaker_feat.shape == (1, strategy.output_dim), \
                f"Speaker feature shape mismatch: {speaker_feat.shape}"
            print(f"Speaker feature extracted successfully. Shape: {speaker_feat.shape}")

        # Test extract_non_speaker
        print("\n--- Testing Non-Speaker Feature Extraction ---")
        # Try to find a non-speaker who might have spoken in the dialogue
        test_non_speaker_name = "Monica" # Example, assume Monica might be a non-speaker in some utterance
        test_dialogue_id = test_utterance.dialogue_id # Use the same dialogue for context

        # This is a basic test. In a real scenario, we'd pick a non-speaker that
        # we expect to have context videos or fallback to zero.
        non_speaker_feat = strategy.extract_non_speaker(test_non_speaker_name, test_dialogue_id)
        assert isinstance(non_speaker_feat, torch.Tensor), "Non-speaker feature is not a Tensor."
        assert non_speaker_feat.shape == (1, strategy.output_dim), \
            f"Non-speaker feature shape mismatch: {non_speaker_feat.shape}"
        print(f"Non-speaker feature extracted successfully. Shape: {non_speaker_feat.shape}")

        print("\n--- VisualCLIPStrategy Self-Test Passed! ---")

    except Exception as e:
        print(f"--- VisualCLIPStrategy Self-Test Failed: {e} ---")
        sys.exit(1)
