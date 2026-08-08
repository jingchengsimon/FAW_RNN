"""Collect real gate/act data for all 9 recurrent-sector contexts (0-8).

Same single-pass-over-the-test-set strategy as gawf_recurrent_gate_multi_digit_collect.py, but
buckets frames by SECTOR label (labels[:, time_idx, 1], per MC_RNN_Dataset's
labels_sector = stack([char_id, sector])) instead of digit, and uses the recurrent-sector
tuning/eligibility from part1_selectivity.npz (primary_hidden_tuning_sector /
primary_hidden_passed_sector) rather than the digit ones.

Writes sector{s}_gate_act_cache.npz (same W/gate/act/T_new/T_old schema as the digit caches)
into the same shared cache directory, so gawf_recurrent_gate_raw_group_sign_grid.py's
load_data(sector, kind="sector") can read them directly. Sectors 1-4 were collected first (to
keep the initial distribution figures cheap); this pass re-covers 1-4 and adds 0, 5-8 so any
sector-pooled analysis (e.g. gawf_recurrent_gate_sign_vs_magnitude_disinhibition_sector.py) can
use the full 9-sector set instead of just 1-4.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SECTORS = tuple(range(9))
MAX_SAMPLES_PER_SECTOR = 1200
TOP_FRACTION = 0.10
SECTOR_LABEL_COL = 1  # labels[:, t, 0]=digit, labels[:, t, 1]=sector



def parse_args() -> argparse.Namespace:
    """Parse one checkpoint's selectivity artifact and isolated output cache."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--selectivity", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "source" / "clutter" / "stimuli"))
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from utils.analysis.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils.analysis.clutter.fig7_relevance_timing import _gate_tensors
    from utils.analysis.clutter.fig7_relevance_stats import relevance_masks

    cli = parse_args()
    dataset_args = argparse.Namespace(
        data_dir=str(cli.data_dir), data_suffix=cli.data_suffix, use_mmap=True,
        chan_num=2, use_sector_mode=True, predict_all_chars=False,
    )
    cli.cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cli.device, require_cuda_if_requested=True)
    dataset, num_pos = build_eval_dataset(dataset_args, "test")
    model = build_model_from_ckpt(str(cli.ckpt), num_pos, device, chan_num=2)
    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.rnn.input_size)
    print(f"dataset sequences={len(dataset)}, hidden_size={hidden_size}, "
          f"sectors={SECTORS}, cap/sector={MAX_SAMPLES_PER_SECTOR}", flush=True)

    W = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)

    with np.load(cli.selectivity, allow_pickle=False) as sel:
        tuning_sector = np.asarray(sel["primary_hidden_tuning_sector"], dtype=np.float64)
        passed_sector = np.asarray(sel["primary_hidden_passed_sector"], dtype=bool)
        dominant = np.asarray(sel["primary_hidden_interaction_dominant"], dtype=bool)
    eligible_sector = passed_sector & ~dominant
    T_old_all = relevance_masks(tuning_sector, eligible_sector, TOP_FRACTION)  # (9, H)
    print(f"eligible (sector)={int(eligible_sector.sum())}  "
          f"T_old sizes per sector: {[int(T_old_all[s].sum()) for s in SECTORS]}", flush=True)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    gate_chunks: dict[int, list[np.ndarray]] = {s: [] for s in SECTORS}
    act_chunks: dict[int, list[np.ndarray]] = {s: [] for s in SECTORS}
    counts: dict[int, int] = {s: 0 for s in SECTORS}
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            labels = labels.to(device=device, dtype=torch.int64, non_blocking=device.type == "cuda")
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            batch_size_actual, frame_num, _ = encoded.shape
            hidden = torch.zeros(batch_size_actual, hidden_size, dtype=encoded.dtype, device=device)
            feedback = torch.zeros(batch_size_actual, model.feedback_dim, dtype=torch.float32, device=device)
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                sector_labels = labels[:, time_idx, SECTOR_LABEL_COL]
                for s in SECTORS:
                    if counts[s] >= MAX_SAMPLES_PER_SECTOR:
                        continue
                    mask = sector_labels == s
                    if mask.any():
                        gate_chunks[s].append(gate_recurrent[mask].detach().cpu().numpy().astype(np.float32))
                        act_chunks[s].append(hidden[mask].detach().cpu().numpy().astype(np.float32))
                        counts[s] += int(mask.sum())
                input_term = torch.einsum(
                    "bi,bhi,hi->bh", encoded[:, time_idx], gate_input, model.rnn.weight_ih_l0
                )
                recurrent_term = torch.einsum(
                    "bi,bhi,hi->bh", hidden, gate_recurrent, model.rnn.weight_hh_l0
                )
                preactivation = input_term + recurrent_term
                if model.rnn.bias_ih_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
                if model.rnn.bias_hh_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
                hidden = torch.relu(model.LNormRNN(torch.tanh(preactivation)))
                char_logits, sector_logits = model.classifier(hidden)
                feedback = torch.cat([char_logits, sector_logits], dim=-1).to(torch.float32)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader) or \
                    all(counts[s] >= MAX_SAMPLES_PER_SECTOR for s in SECTORS):
                print(f"  batch {batch_idx + 1}/{len(loader)}  counts={counts}  "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            if all(counts[s] >= MAX_SAMPLES_PER_SECTOR for s in SECTORS):
                print("  all sector caps reached, stopping early", flush=True)
                break

    for s in SECTORS:
        if not gate_chunks[s]:
            raise RuntimeError(f"sector {s}: no frames collected")
        G_flat = np.concatenate(gate_chunks[s], axis=0)[:MAX_SAMPLES_PER_SECTOR]
        act_flat = np.concatenate(act_chunks[s], axis=0)[:MAX_SAMPLES_PER_SECTOR]
        n_samples = G_flat.shape[0]
        print(f"sector {s}: collected {n_samples} samples", flush=True)

        top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
        T_new = np.zeros(hidden_size, dtype=bool)
        T_new[np.argsort(-act_flat.mean(axis=0))[:top_k]] = True

        gate = G_flat.reshape(1, n_samples, 1, hidden_size, hidden_size)
        act = act_flat.reshape(1, n_samples, 1, hidden_size)

        cache_path = cli.cache_dir / f"sector{s}_gate_act_cache.npz"
        np.savez(cache_path, W=W.astype(np.float32), gate=gate, act=act,
                 T_new=T_new, T_old=T_old_all[s])
        print(f"Saved {cache_path}", flush=True)


if __name__ == "__main__":
    main()
