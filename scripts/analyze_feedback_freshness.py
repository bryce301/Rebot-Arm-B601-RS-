#!/usr/bin/env python3
"""Summarize timestamped RobStride feedback freshness in a recorded NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="NPZ written by rebot_record_teleop.py or rebot_replay.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file).expanduser()
    data = np.load(path, allow_pickle=True)

    age_key = "rs_feedback_age_s" if "rs_feedback_age_s" in data else "feedback_age_s"
    sequence_key = "rs_feedback_sequence" if "rs_feedback_sequence" in data else "feedback_sequence"
    source_key = "rs_feedback_source" if "rs_feedback_source" in data else "feedback_source"
    receipt_key = (
        "rs_feedback_received_monotonic_s"
        if "rs_feedback_received_monotonic_s" in data
        else "feedback_received_monotonic_s"
    )
    required = (age_key, sequence_key, source_key, receipt_key, "joint_names")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"{path} does not contain timestamped feedback fields: {', '.join(missing)}"
        )

    joint_names = [str(name) for name in data["joint_names"]]
    ages_ms = np.asarray(data[age_key], dtype=np.float64) * 1000.0
    sequences = np.asarray(data[sequence_key], dtype=np.int64)
    sources = np.asarray(data[source_key]).astype(str)
    receipt_s = np.asarray(data[receipt_key], dtype=np.float64)
    if ages_ms.ndim != 2 or ages_ms.shape[1] != len(joint_names):
        raise ValueError(f"unexpected feedback shape: {ages_ms.shape}")

    print(f"File: {path}")
    print(f"Frames: {ages_ms.shape[0]}")
    print("Per-joint feedback freshness:")
    for index, name in enumerate(joint_names):
        finite_age = ages_ms[:, index][np.isfinite(ages_ms[:, index])]
        valid_sequence = sequences[:, index][sequences[:, index] >= 0]
        repeated = (
            float(np.mean(np.diff(valid_sequence) == 0)) * 100.0
            if valid_sequence.size > 1
            else float("nan")
        )
        source_values, source_counts = np.unique(sources[:, index], return_counts=True)
        source_summary = ", ".join(
            f"{source}={count / len(sources[:, index]) * 100.0:.1f}%"
            for source, count in zip(source_values, source_counts, strict=True)
        )
        if finite_age.size == 0:
            print(f"  {name:14s} no timestamped feedback")
            continue
        print(
            f"  {name:14s} mean={np.mean(finite_age):6.2f}ms "
            f"p95={np.percentile(finite_age, 95):6.2f}ms "
            f"max={np.max(finite_age):6.2f}ms repeated_seq={repeated:5.1f}% "
            f"{source_summary}"
        )

    valid_rows = np.all(np.isfinite(receipt_s), axis=1)
    skew_ms = np.ptp(receipt_s[valid_rows], axis=1) * 1000.0
    if skew_ms.size:
        print(
            "Frame receipt skew: "
            f"mean={np.mean(skew_ms):.2f}ms "
            f"p95={np.percentile(skew_ms, 95):.2f}ms "
            f"max={np.max(skew_ms):.2f}ms"
        )


if __name__ == "__main__":
    main()
