import argparse
import sys  # noqa: E402
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AppConfig  # noqa: E402
from src.parser import parse_dev_txt, DialogueRecord  # noqa: E402
from src.extractors.visual_clip.strategy import VisualCLIPStrategy  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Dialogue Feature Extraction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: only process the first --max-dialogues dialogues (default 2).",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=2,
        metavar="N",
        help="Number of dialogues to process in smoke-test mode (default: 2).",
    )
    args = parser.parse_args()

    # --- 加载配置 ---
    try:
        app_config = AppConfig.from_yaml(args.config)
        print(f"[main] Configuration loaded from {args.config}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[main] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[main] Unexpected error while loading config: {e}")
        sys.exit(1)

    # --- 解析 dev.txt ---
    print(f"[main] Parsing dev.txt: {app_config.paths.dev_txt}")
    try:
        dialogue_records: List[DialogueRecord] = parse_dev_txt(app_config.paths.dev_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[main] Parse error: {e}")
        sys.exit(1)
    print(f"[main] Parsed {len(dialogue_records)} dialogues in total.")

    # --- Smoke-test 截断 ---
    if args.smoke:
        n = min(args.max_dialogues, len(dialogue_records))
        dialogue_records = dialogue_records[:n]
        print(
            f"[main] *** SMOKE-TEST MODE: processing only {n} dialogue(s) "
            f"({sum(len(d.utterances) for d in dialogue_records)} utterances) ***"
        )

    # --- 初始化提取器 ---
    if app_config.extractor.active_type == "visual_clip":
        if app_config.extractor.visual_clip_config is None:
            print("[main] Error: visual_clip_config is missing in config.yaml.")
            sys.exit(1)
        print("[main] Initializing VisualCLIPStrategy...")
        extractor = VisualCLIPStrategy(app_config.extractor.visual_clip_config, app_config.non_speaker)
    else:
        print(f"[main] Unsupported extractor type: {app_config.extractor.active_type}")
        sys.exit(1)
    print(f"[main] Extractor ready, output_dim={extractor.output_dim}")

    # --- 确保输出目录存在 ---
    app_config.paths.feat_out.mkdir(parents=True, exist_ok=True)

    # --- 运行 Pipeline ---
    run_pipeline(app_config, extractor, dialogue_records)

    if args.smoke:
        print("[main] Smoke-test finished successfully.")


if __name__ == "__main__":
    main()
