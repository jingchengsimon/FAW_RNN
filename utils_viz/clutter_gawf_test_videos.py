"""Render annotated Clutter test-set sequences for a single-layer GaWF checkpoint.

Inputs are one canonical Clutter test split and a checkpoint.  Each output MP4 shows every
time step's predicted sector and digit, while a circle identifies the ground-truth foreground
digit.  A JSON manifest records samples, labels, predictions, and rendering settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.clutter_data_pipeline import prepare_clutter_inputs
from utils.clutter_train_acceleration import run_forward_with_feedback
from utils_viz.generate_clutter_context_demo import _write_video


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, test-data, and presentation-video arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def _centered_text(image: np.ndarray, text: str, y: int, scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(
        image,
        text,
        ((image.shape[1] - width) // 2, y),
        font,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )


def _render_frame(
    stimulus: np.ndarray,
    time_step: int,
    total_steps: int,
    predicted_digit: int,
    predicted_sector: int,
    true_digit: int,
    true_sector: int,
    true_x: float,
    true_y: float,
    scale: int,
) -> np.ndarray:
    """Create one presentation frame with model predictions and a ground-truth circle."""
    image = cv2.resize(
        stimulus.astype(np.uint8),
        (stimulus.shape[1] * scale, stimulus.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    header, footer = 54, 70
    panel = np.full((header + image.shape[0] + footer, image.shape[1], 3), 20, dtype=np.uint8)
    panel[header : header + image.shape[0]] = image
    title = f"GaWF L1 test sequence | time step {time_step + 1}/{total_steps}"
    _centered_text(panel, title, 35, 0.67)
    center = (int(round(true_x * scale)), header + int(round(true_y * scale)))
    cv2.circle(panel, center, int(round(17 * scale)), (70, 210, 90), max(2, scale), cv2.LINE_AA)
    _centered_text(
        panel,
        f"prediction: sector {predicted_sector}, digit {predicted_digit}",
        header + image.shape[0] + 28,
        0.55,
        (235, 235, 235),
    )
    _centered_text(
        panel,
        f"ground truth: sector {true_sector}, digit {true_digit}  (circled)",
        header + image.shape[0] + 56,
        0.55,
        (70, 210, 90),
    )
    return panel


def main() -> None:
    """Run model inference on selected test sequences and write annotated MP4 videos."""
    args = parse_args()
    if args.fps <= 0 or args.scale < 1 or len(args.sample_indices) < 4:
        raise ValueError("Require fps > 0, scale >= 1, and at least four sample indices")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset, num_pos = build_test_dataset(args)
    model = build_model_from_ckpt(args.checkpoint, num_pos, device, chan_num=args.chan_num)
    if int(getattr(model, "num_layers", 1)) != 1 or not getattr(model, "is_gawf_model", False):
        raise ValueError("--checkpoint must be a single-layer GaWF model")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": "test",
        "data_suffix": args.data_suffix,
        "sample_indices": args.sample_indices,
        "fps": args.fps,
        "scale": args.scale,
        "videos": [],
    }
    for sample_index in args.sample_indices:
        if not 0 <= sample_index < len(dataset):
            raise IndexError(f"sample index out of range: {sample_index}")
        frames, labels = dataset[sample_index][:2]
        inputs = prepare_clutter_inputs(
            torch.as_tensor(frames).unsqueeze(0),
            device=device,
            cast_mode="device",
            frame_layout="stacked",
            chan_num=args.chan_num,
        )
        with torch.no_grad():
            char_logits, sector_logits = run_forward_with_feedback(model, inputs, use_feedback=True)
        pred_digits = char_logits.argmax(dim=2).squeeze(0).cpu().tolist()
        pred_sectors = sector_logits.argmax(dim=2).squeeze(0).cpu().tolist()
        label_array = np.asarray(labels, dtype=np.int64)
        start = sample_index * dataset.frame_num + dataset.chan_num
        coordinates = dataset.labels[start : start + dataset.frame_num, 1:3]
        visual_frames = []
        for time_index in range(dataset.frame_num):
            stimulus = np.asarray(frames[time_index, -1], dtype=np.uint8)
            visual_frames.append(
                _render_frame(
                    stimulus, time_index, dataset.frame_num, int(pred_digits[time_index]),
                    int(pred_sectors[time_index]), int(label_array[time_index, 0]),
                    int(label_array[time_index, 1]), float(coordinates[time_index, 0]),
                    float(coordinates[time_index, 1]), args.scale,
                )
            )
        output_path = output_dir / f"clutter_gawf_l1_test_sample{sample_index}.mp4"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {output_path}")
        _write_video(output_path, visual_frames, args.fps)
        manifest["videos"].append(
            {
                "sample_index": sample_index,
                "path": str(output_path),
                "ground_truth_digits": label_array[:, 0].tolist(),
                "ground_truth_sectors": label_array[:, 1].tolist(),
                "predicted_digits": pred_digits,
                "predicted_sectors": pred_sectors,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(args.sample_indices)} Clutter GaWF test videos to {output_dir}")


if __name__ == "__main__":
    main()
