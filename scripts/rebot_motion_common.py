#!/usr/bin/env python3
"""Shared helpers for B601-RS recording and replay scripts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]

# Conservative intersection of Seeed LeRobot limits and the B601-RS URDF limits.
# Values are actual RS joint coordinates, not leader/action coordinates.
CONSERVATIVE_LIMITS_DEG = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (0.0, 170.0),
    "elbow_flex": (0.0, 179.9),
    "wrist_flex": (-80.0, 89.9),
    "wrist_yaw": (-89.9, 89.9),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (0.0, 265.0),
}


def conservative_limits_rad(joint_names: list[str] | tuple[str, ...] = JOINT_NAMES) -> np.ndarray:
    return np.array(
        [[math.radians(lo), math.radians(hi)] for lo, hi in (CONSERVATIVE_LIMITS_DEG[name] for name in joint_names)],
        dtype=np.float64,
    )


def clamp_positions_rad(pos_rad: np.ndarray, limits_rad: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_rad, dtype=np.float64).copy()
    return np.clip(pos, limits_rad[:, 0], limits_rad[:, 1])


def minimum_jerk(u: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def sleep_to_next_tick(loop_start: float, period_s: float) -> float:
    elapsed = time.perf_counter() - loop_start
    remaining = period_s - elapsed
    if remaining > 0:
        time.sleep(remaining)
    return time.perf_counter() - loop_start


def dict_to_vector(values: dict[str, float], joint_names: list[str]) -> np.ndarray:
    return np.array([float(values[name]) for name in joint_names], dtype=np.float64)


def leader_action_to_rs_target_rad(
    leader_action_deg: dict[str, float],
    joint_names: list[str],
    joint_directions: dict[str, float],
    limits_rad: np.ndarray,
) -> np.ndarray:
    target_deg = []
    for name in joint_names:
        key = f"{name}.pos"
        if key not in leader_action_deg:
            raise KeyError(f"leader action missing {key}")
        direction = float(joint_directions[name])
        target_deg.append(float(leader_action_deg[key]) * direction)
    return clamp_positions_rad(np.radians(np.asarray(target_deg, dtype=np.float64)), limits_rad)


def request_and_read_states(robot: Any, joint_names: list[str]) -> dict[str, Any | None]:
    get_motor_states = getattr(robot, "get_motor_states", None)
    if callable(get_motor_states):
        states = get_motor_states()
        return {name: states.get(name) for name in joint_names}

    for motor in robot.motors.values():
        try:
            motor.request_feedback()
        except Exception:
            pass
    try:
        robot.bus.poll_feedback_once()
    except Exception:
        pass
    return {name: robot.motors[name].get_state() for name in joint_names}


def states_to_arrays(
    states: dict[str, Any | None],
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.full(len(joint_names), np.nan, dtype=np.float64)
    vel = np.full(len(joint_names), np.nan, dtype=np.float64)
    torque = np.full(len(joint_names), np.nan, dtype=np.float64)
    status = np.full(len(joint_names), -1, dtype=np.int32)
    for i, name in enumerate(joint_names):
        state = states.get(name)
        if state is None:
            continue
        pos[i] = float(state.pos)
        vel[i] = float(state.vel)
        torque[i] = float(state.torq)
        status[i] = int(state.status_code)
    return pos, vel, torque, status


def states_to_timing_arrays(
    states: dict[str, Any | None],
    joint_names: list[str],
    captured_monotonic_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    captured_s = time.monotonic() if captured_monotonic_s is None else captured_monotonic_s
    age_s = np.full(len(joint_names), np.nan, dtype=np.float64)
    received_monotonic_s = np.full(len(joint_names), np.nan, dtype=np.float64)
    received_wall_s = np.full(len(joint_names), np.nan, dtype=np.float64)
    sequence = np.full(len(joint_names), -1, dtype=np.int64)
    source = np.full(len(joint_names), "unknown", dtype="<U20")
    for i, name in enumerate(joint_names):
        state = states.get(name)
        if state is None:
            continue
        received_s = getattr(state, "received_monotonic_s", None)
        if received_s is not None:
            received_monotonic_s[i] = float(received_s)
            age_s[i] = max(0.0, captured_s - float(received_s))
        wall_s = getattr(state, "received_wall_s", None)
        if wall_s is not None:
            received_wall_s[i] = float(wall_s)
        state_sequence = getattr(state, "sequence", None)
        if state_sequence is not None:
            sequence[i] = int(state_sequence)
        state_source = getattr(state, "source", None)
        if state_source is not None:
            source[i] = str(state_source)
    return age_s, received_monotonic_s, received_wall_s, sequence, source


def current_positions_rad(robot: Any, joint_names: list[str], limits_rad: np.ndarray) -> np.ndarray:
    states = request_and_read_states(robot, joint_names)
    pos, _, _, _ = states_to_arrays(states, joint_names)
    if not np.all(np.isfinite(pos)):
        missing = [joint_names[i] for i, ok in enumerate(np.isfinite(pos)) if not ok]
        raise RuntimeError(f"missing RS feedback for joints: {missing}")
    return clamp_positions_rad(pos, limits_rad)


@dataclass
class GripperTorqueLimiter:
    kp: float
    kd: float
    torque_limit: float
    hold_torque_limit: float
    motion_velocity_threshold: float = 0.25
    target_vel_max: float = 3.0
    lpf_alpha: float = 0.3
    prev_target_pos: float | None = None
    prev_filtered_target_vel: float | None = None
    prev_state_pos: float | None = None

    def compute(self, target_pos: float, state: Any | None, dt_s: float) -> float:
        if state is None or dt_s <= 0.0:
            return 0.0
        if self.prev_target_pos is None:
            target_vel = 0.0
        else:
            target_vel = (target_pos - self.prev_target_pos) / dt_s
        self.prev_target_pos = target_pos

        if self.prev_filtered_target_vel is None:
            filtered_target_vel = target_vel
        else:
            filtered_target_vel = self.lpf_alpha * target_vel + (1.0 - self.lpf_alpha) * self.prev_filtered_target_vel
        target_vel = float(np.clip(filtered_target_vel, -self.target_vel_max, self.target_vel_max))
        self.prev_filtered_target_vel = target_vel

        state_pos = float(state.pos)
        if self.prev_state_pos is None:
            estimated_state_vel = 0.0
        else:
            estimated_state_vel = (state_pos - self.prev_state_pos) / dt_s
        self.prev_state_pos = state_pos

        tau = self.kp * (target_pos - state_pos) + self.kd * (target_vel - float(state.vel))
        if abs(estimated_state_vel) > self.motion_velocity_threshold:
            limit = max(0.0, float(self.torque_limit))
        else:
            limit = max(0.0, float(self.hold_torque_limit))
        return float(np.clip(tau, -limit, limit))


def send_rs_mit_targets(
    robot: Any,
    target_pos_rad: np.ndarray,
    joint_names: list[str],
    states: dict[str, Any | None],
    gripper_limiter: GripperTorqueLimiter,
    dt_s: float,
) -> None:
    target = np.asarray(target_pos_rad, dtype=np.float64)
    for i, name in enumerate(joint_names):
        motor = robot.motors.get(name)
        if motor is None:
            continue
        if name == "gripper":
            control_mode = str(getattr(robot.config, "gripper_control_mode", "torque"))
            if control_mode == "position_mit":
                kp = float(robot.config.gripper_position_mit_kp)
                kd = float(robot.config.gripper_position_mit_kd)
                motor.send_mit(float(target[i]), 0.0, kp, kd, 0.0)
            elif control_mode == "torque":
                tau = gripper_limiter.compute(float(target[i]), states.get(name), dt_s)
                motor.send_mit(0.0, 0.0, 0.0, 1.5, tau)
            else:
                raise ValueError(f"unsupported gripper_control_mode={control_mode!r}")
            continue
        kp = float(robot.config.mit_kp.get(name, 0.0))
        kd = float(robot.config.mit_kd.get(name, 0.0))
        motor.send_mit(float(target[i]), 0.0, kp, kd, 0.0)


def move_smoothly(
    robot: Any,
    start_rad: np.ndarray,
    goal_rad: np.ndarray,
    duration_s: float,
    fps: int,
    joint_names: list[str],
    limits_rad: np.ndarray,
    gripper_limiter: GripperTorqueLimiter,
    label: str,
) -> None:
    if duration_s <= 0:
        states = request_and_read_states(robot, joint_names)
        send_rs_mit_targets(robot, clamp_positions_rad(goal_rad, limits_rad), joint_names, states, gripper_limiter, 1.0 / fps)
        return

    period_s = 1.0 / fps
    steps = max(2, int(round(duration_s * fps)))
    start = clamp_positions_rad(start_rad, limits_rad)
    goal = clamp_positions_rad(goal_rad, limits_rad)
    print(f"{label}: {duration_s:.2f}s, {steps} steps")
    for step in range(steps):
        loop_start = time.perf_counter()
        u = step / (steps - 1)
        s = minimum_jerk(u)
        target = start + s * (goal - start)
        states = request_and_read_states(robot, joint_names)
        send_rs_mit_targets(robot, target, joint_names, states, gripper_limiter, period_s)
        sleep_to_next_tick(loop_start, period_s)


def prompt_before_motion(message: str, assume_yes: bool = False) -> None:
    if assume_yes:
        return
    answer = input(f"{message}\nPress ENTER to continue, or type q then ENTER to abort: ").strip().lower()
    if answer in {"q", "quit", "exit", "n", "no"}:
        raise RuntimeError("aborted by user before motion")


def make_metadata(**kwargs: Any) -> np.ndarray:
    return np.array(json.dumps(kwargs, ensure_ascii=False, indent=2), dtype=object)


def save_npz(path: str | Path, **arrays: Any) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)


def limit_relative_steps(trajectory: np.ndarray, max_step_rad: float | None) -> tuple[np.ndarray, int]:
    if max_step_rad is None or max_step_rad <= 0:
        return trajectory, 0
    traj = np.asarray(trajectory, dtype=np.float64).copy()
    clipped_count = 0
    for i in range(1, traj.shape[0]):
        delta = traj[i] - traj[i - 1]
        clipped_delta = np.clip(delta, -max_step_rad, max_step_rad)
        clipped_count += int(np.count_nonzero(clipped_delta != delta))
        traj[i] = traj[i - 1] + clipped_delta
    return traj, clipped_count


def load_and_clean_trajectory(
    path: str | Path,
    source: str,
    limits_rad: np.ndarray,
    allow_fill_missing: bool = False,
    max_step_rad: float | None = None,
) -> dict[str, Any]:
    data = np.load(Path(path).expanduser(), allow_pickle=True)
    if source not in data:
        raise KeyError(f"{source!r} not found in {path}")
    traj = np.asarray(data[source], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"{source} must have shape (N, {len(JOINT_NAMES)}), got {traj.shape}")

    if not np.all(np.isfinite(traj)):
        bad_rows = np.flatnonzero(~np.all(np.isfinite(traj), axis=1))
        if not allow_fill_missing:
            preview = ", ".join(map(str, bad_rows[:10]))
            raise ValueError(
                f"{source} contains non-finite values in {len(bad_rows)} rows "
                f"(first rows: {preview}). Re-record, or use --allow-fill-missing."
            )
        print("[warn] trajectory contains non-finite values; filling from previous valid sample")
        for i in range(traj.shape[0]):
            if i == 0:
                traj[i] = np.nan_to_num(traj[i], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                bad = ~np.isfinite(traj[i])
                traj[i, bad] = traj[i - 1, bad]
    traj = clamp_positions_rad(traj, limits_rad)
    traj, clipped_count = limit_relative_steps(traj, max_step_rad)
    if clipped_count:
        print(f"[warn] clipped {clipped_count} per-joint trajectory jumps by max_step_rad={max_step_rad}")
    traj = clamp_positions_rad(traj, limits_rad)
    return {"data": data, "trajectory": traj, "clipped_jump_count": clipped_count}
