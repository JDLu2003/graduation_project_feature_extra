#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


FPS = 25.0
MIN_FACE_CONFIDENCE = 0.95


@dataclass(frozen=True)
class FrameDetection:
    frame_name: str
    boxes: list[list[float]]
    probs: list[float]


def _require_absolute_path(video_path: Path) -> None:
    assert video_path.is_absolute(), f"输入必须是视频绝对路径，当前得到: {video_path}"
    assert video_path.exists(), f"视频不存在: {video_path}"
    assert video_path.is_file(), f"输入不是文件: {video_path}"


def _require_ffmpeg() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise FileNotFoundError("未找到 ffmpeg，请先安装 ffmpeg 并确保其在 PATH 中。")
    return ffmpeg_bin


def _load_mtcnn(device: str):
    try:
        from facenet_pytorch import MTCNN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未安装 facenet_pytorch，无法执行 MTCNN 人脸检测。"
        ) from exc

    return MTCNN(
        image_size=160,
        margin=12,
        min_face_size=24,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=True,
        device=device,
    )


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_frames(ffmpeg_bin: str, video_path: Path, frame_dir: Path, fps: float) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%06d.png"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(pattern),
    ]
    _run_command(cmd)


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_boxes(image: Image.Image, boxes: np.ndarray | None, probs: np.ndarray | None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    if boxes is None or probs is None:
        return image

    for idx, (box, prob) in enumerate(zip(boxes, probs), start=1):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=max(2, image.width // 200))
        label = f"face{idx}:{float(prob):.2f}"
        text_x = x1
        text_y = max(0.0, y1 - 18.0)
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 0, 0))
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
    return image


def _detect_faces_on_frames(
    mtcnn,
    frame_dir: Path,
    boxed_dir: Path,
    min_confidence: float,
) -> list[FrameDetection]:
    boxed_dir.mkdir(parents=True, exist_ok=True)
    detections: list[FrameDetection] = []

    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"未在帧目录中找到抽帧结果: {frame_dir}")

    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            rgb = img.convert("RGB")
            boxes, probs = mtcnn.detect(rgb)
            if boxes is not None and probs is not None:
                probs_np = np.asarray(probs, dtype=float)
                boxes_np = np.asarray(boxes, dtype=float)
                keep = probs_np >= min_confidence
                boxes = boxes_np[keep]
                probs = probs_np[keep]
                if len(boxes) == 0:
                    boxes = None
                    probs = None
            annotated = _draw_boxes(rgb.copy(), boxes, probs)
            out_path = boxed_dir / frame_path.name
            annotated.save(out_path)

            detections.append(
                FrameDetection(
                    frame_name=frame_path.name,
                    boxes=[] if boxes is None else boxes.astype(float).tolist(),
                    probs=[] if probs is None else [float(x) for x in probs.tolist()],
                )
            )

    return detections


def _rebuild_video(ffmpeg_bin: str, boxed_dir: Path, source_video: Path, output_video: Path, fps: float) -> None:
    pattern = boxed_dir / "frame_%06d.png"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
    ]
    cmd.append(str(output_video))
    _run_command(cmd)


def _write_manifest(manifest_path: Path, source_video: Path, detections: Iterable[FrameDetection], boxed_video: Path) -> None:
    payload = {
        "source_video": str(source_video),
        "boxed_video": str(boxed_video),
        "fps": FPS,
        "frames": [asdict(item) for item in detections],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 ffmpeg 抽帧并用 MTCNN 检测人脸，生成带框视频。")
    parser.add_argument("video_path", type=Path, help="输入视频的绝对路径")
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="抽帧与重建视频使用的帧率，默认 25.0（即 0.04s 一帧）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="MTCNN 运行设备，默认 cpu。",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_FACE_CONFIDENCE,
        help="人脸置信度阈值，默认 0.95。",
    )
    args = parser.parse_args()

    video_path = args.video_path.expanduser().resolve()
    _require_absolute_path(video_path)
    ffmpeg_bin = _require_ffmpeg()

    output_root = project_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{video_path.stem}_mtcnn_{_timestamp()}"
    frame_dir = run_dir / "frames"
    boxed_dir = run_dir / "boxed_frames"
    output_video = run_dir / f"{video_path.stem}_boxed.mp4"
    manifest_path = run_dir / "manifest.json"

    print(f"[extract_faces_with_ffmpeg_mtcnn] 输入视频: {video_path}")
    print(f"[extract_faces_with_ffmpeg_mtcnn] 输出目录: {run_dir}")

    try:
        mtcnn = _load_mtcnn(args.device)

        print("[extract_faces_with_ffmpeg_mtcnn] Step 1/3: 使用 ffmpeg 抽帧...")
        _extract_frames(ffmpeg_bin, video_path, frame_dir, args.fps)

        print("[extract_faces_with_ffmpeg_mtcnn] Step 2/3: 使用 MTCNN 检测人脸并保存框图...")
        detections = _detect_faces_on_frames(mtcnn, frame_dir, boxed_dir, args.min_confidence)

        print("[extract_faces_with_ffmpeg_mtcnn] Step 3/3: 将框图重建为视频...")
        _rebuild_video(ffmpeg_bin, boxed_dir, video_path, output_video, args.fps)

        _write_manifest(manifest_path, video_path, detections, output_video)

        print(f"[extract_faces_with_ffmpeg_mtcnn] 完成: {output_video}")
        print(f"[extract_faces_with_ffmpeg_mtcnn] 帧图目录: {frame_dir}")
        print(f"[extract_faces_with_ffmpeg_mtcnn] 框图目录: {boxed_dir}")
        print(f"[extract_faces_with_ffmpeg_mtcnn] 记录文件: {manifest_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[extract_faces_with_ffmpeg_mtcnn] ffmpeg 运行失败: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[extract_faces_with_ffmpeg_mtcnn] 运行失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
