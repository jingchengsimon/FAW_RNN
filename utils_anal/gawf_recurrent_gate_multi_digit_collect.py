"""Collect real gate/act data for digits 1-9 in a single pass (digit 0 already cached).

Companion to gawf_recurrent_gate_single_digit_diagnostic_collect.py, which collects the
exhaustive digit-0 cache (all 6437 frames, used to verify TT≈0.238) in one full pass over the
test set. Re-running that script 9 more times would redo the same ~254s forward pass 9 times
just to keep a different digit's frames -- wasteful, since every frame's gate/hidden is already
computed regardless of which digit it belongs to. This script instead does ONE pass and buckets
frames by digit for digits 1-9, capping each digit at MAX_SAMPLES_PER_DIGIT to keep peak memory
bounded (this machine has 8GB RAM; 9 digits x ~1200 samples x 256x256 float32 stays under 3GB).

Writes the same digit{d}_gate_act_cache.npz schema (W, gate, act, T_new, T_old) as the digit-0
collector, into the same output directory, so gawf_recurrent_gate_raw_group_sign_grid.py's
generalized load_data(digit=d) can read any of them interchangeably.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir

CACHE_CATEGORY = "E_relevance_alignment"
CACHE_SCRIPT_NAME = "gawf_recurrent_gate_single_digit_diagnostic_collect"  # shared cache dir
DIGITS = tuple(range(1, 10))  # digit 0 already collected separately
MAX_SAMPLES_PER_DIGIT = 1200
TOP_FRACTION = 0.10

CKPT_PATH = PROJECT_ROOT / (
    "results/data/train_data/clutter/best_6model_param_matched_40h/"
    "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
)
SELECTIVITY_PATH = PROJECT_ROOT / (
    "results/data/anal_data/D_variance_decomposition/"
    "gawf_symmetric_relevance_timing/part1_selectivity.npz"
)
DATA_DIR = PROJECT_ROOT / "stimuli"
DATA_SUFFIX = "40h-uint8"
EXPECTED_SHA256 = "5b109ff973bc54726be5a58cb50699d3e150da9d521da0a1126c089a2aeefc25"


def main() -> None:
    import hashlib

    import torch
    from torch.utils.data import DataLoader

    from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils_anal.gawf_symmetric_relevance_timing import _gate_tensors
    from utils_anal.gawf_symmetric_stats import relevance_masks

    checksum = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise RuntimeError(f"checkpoint sha256 mismatch: {checksum} != {EXPECTED_SHA256}")

    args = argparse.Namespace(
        data_dir=str(DATA_DIR), data_suffix=DATA_SUFFIX, use_mmap=True,
        chan_num=2, use_sector_mode=True, predict_all_chars=False,
    )
    device = resolve_device("cpu")
    dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(CKPT_PATH), num_pos, device, chan_num=2)
    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.rnn.input_size)
    print(f"dataset sequences={len(dataset)}, hidden_size={hidden_size}, "
          f"digits={DIGITS}, cap/digit={MAX_SAMPLES_PER_DIGIT}", flush=True)

    W = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)

    with np.load(SELECTIVITY_PATH, allow_pickle=False) as sel:
        tuning = np.asarray(sel["primary_hidden_tuning_digit"], dtype=np.float64)
        passed = np.asarray(sel["primary_hidden_passed_digit"], dtype=bool)
        dominant = np.asarray(sel["primary_hidden_interaction_dominant"], dtype=bool)
    eligible = passed & ~dominant
    T_old_all = relevance_masks(tuning, eligible, TOP_FRACTION)  # (10, H)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    gate_chunks: dict[int, list[np.ndarray]] = {d: [] for d in DIGITS}
    act_chunks: dict[int, list[np.ndarray]] = {d: [] for d in DIGITS}
    counts: dict[int, int] = {d: 0 for d in DIGITS}
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.int64)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            batch_size_actual, frame_num, _ = encoded.shape
            hidden = torch.zeros(batch_size_actual, hidden_size, dtype=encoded.dtype)
            feedback = torch.zeros(batch_size_actual, model.feedback_dim, dtype=torch.float32)
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                digit_labels = labels[:, time_idx, 0]
                for d in DIGITS:
                    if counts[d] >= MAX_SAMPLES_PER_DIGIT:
                        continue
                    mask = digit_labels == d
                    if mask.any():
                        gate_chunks[d].append(gate_recurrent[mask].detach().cpu().numpy().astype(np.float32))
                        act_chunks[d].append(hidden[mask].detach().cpu().numpy().astype(np.float32))
                        counts[d] += int(mask.sum())
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
                    all(counts[d] >= MAX_SAMPLES_PER_DIGIT for d in DIGITS):
                print(f"  batch {batch_idx + 1}/{len(loader)}  counts={counts}  "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            if all(counts[d] >= MAX_SAMPLES_PER_DIGIT for d in DIGITS):
                print("  all digit caps reached, stopping early", flush=True)
                break

    data_dir_out = output_dir(CACHE_CATEGORY, CACHE_SCRIPT_NAME, "data")
    for d in DIGITS:
        G_flat = np.concatenate(gate_chunks[d], axis=0)[:MAX_SAMPLES_PER_DIGIT]
        act_flat = np.concatenate(act_chunks[d], axis=0)[:MAX_SAMPLES_PER_DIGIT]
        n_samples = G_flat.shape[0]
        print(f"digit {d}: collected {n_samples} samples", flush=True)

        top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
        T_new = np.zeros(hidden_size, dtype=bool)
        T_new[np.argsort(-act_flat.mean(axis=0))[:top_k]] = True

        gate = G_flat.reshape(1, n_samples, 1, hidden_size, hidden_size)
        act = act_flat.reshape(1, n_samples, 1, hidden_size)

        cache_path = data_dir_out / f"digit{d}_gate_act_cache.npz"
        np.savez(cache_path, W=W.astype(np.float32), gate=gate, act=act,
                 T_new=T_new, T_old=T_old_all[d])
        print(f"Saved {cache_path}", flush=True)


if __name__ == "__main__":
    main()
