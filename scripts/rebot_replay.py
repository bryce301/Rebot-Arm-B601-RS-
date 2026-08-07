#!/usr/bin/env python3
"""Replay a recorded B601-RS trajectory with direct RobStride MIT commands."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from lerobot_robot_seeed_b601.seeed_b601_rs_follower import SeeedB601RSFollower

from rebot_motion_common import (
    CONSERVATIVE_LIMITS_DEG,
    GripperTorqueLimiter,
    JOINT_NAMES,
    conservative_limits_rad,
    current_positions_rad,
    limit_gripper_trajectory_steps,
    load_and_clean_trajectory,
    move_smoothly,
    prompt_before_motion,
    request_and_read_states,
    save_npz,
    send_rs_mit_targets,
    sleep_to_next_tick,
    states_to_arrays,
    states_to_timing_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-port", default="can0", help="B601-RS CAN channel")
    parser.add_argument("--can-adapter", default="socketcan", choices=["socketcan", "damiao"])
    parser.add_argument("--file", required=True, help="Recorded .npz file")
    parser.add_argument("--source", default="rs_target_pos_rad", help="Trajectory key in the .npz file")
    parser.add_argument("--fps", type=int, default=150)
    parser.add_argument("--zero-time", type=float, default=5.0, help="Smooth time from current pose to zero pose")
    parser.add_argument("--start-time", type=float, default=3.0, help="Smooth time from zero pose to first recorded frame")
    parser.add_argument("--gripper-kp", type=float, default=12.0)
    parser.add_argument("--gripper-kd", type=float, default=1.9)
    parser.add_argument("--gripper-max-step-deg", type=float, default=3.0)
    parser.add_argument("--gripper-torque-limit", type=float, default=0.5, help="Runtime 0x700B limit in Nm")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="Exclusive end frame. Use -1 for all frames.")
    parser.add_argument("--max-step-rad", type=float, default=0.35, help="Clamp per-frame joint jumps. Use <=0 to disable.")
    parser.add_argument(
        "--final-hold-time",
        type=float,
        default=2.0,
        help="Keep commanding the last frame before disconnecting, while logging tracking error",
    )
    parser.add_argument("--log-out", help="Optional output NPZ for target/actual joint tracking data")
    parser.add_argument("--allow-fill-missing", action="store_true", help="Fill NaN/Inf samples from previous samples instead of aborting")
    parser.add_argument("--robot-id", default="seeed_b601_rs_follower")
    parser.add_argument("--calibrate-robot", action="store_true", help="Allow robot calibration prompt during connect")
    parser.add_argument("--keep-torque-on-exit", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the pre-motion confirmation prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.gripper_kp < 0.0 or args.gripper_kd < 0.0:
        raise ValueError("--gripper-kp and --gripper-kd must be non-negative")
    if args.gripper_torque_limit <= 0.0:
        raise ValueError("--gripper-torque-limit must be positive")
    if args.gripper_max_step_deg <= 0.0:
        raise ValueError("--gripper-max-step-deg must be positive")

    period_s = 1.0 / args.fps
    gripper_max_step_rad = math.radians(args.gripper_max_step_deg)
    limits_rad = conservative_limits_rad(JOINT_NAMES)
    loaded = load_and_clean_trajectory(
        args.file,
        args.source,
        limits_rad,
        allow_fill_missing=args.allow_fill_missing,
        max_step_rad=args.max_step_rad,
    )
    trajectory = loaded["trajectory"]
    trajectory, gripper_clipped_count = limit_gripper_trajectory_steps(
        trajectory,
        JOINT_NAMES,
        gripper_max_step_rad,
    )
    if gripper_clipped_count:
        print(
            f"[warn] limited {gripper_clipped_count} gripper target steps "
            f"to {args.gripper_max_step_deg:g} deg/frame"
        )

    start_frame = max(0, args.start_frame)
    end_frame = trajectory.shape[0] if args.end_frame < 0 else min(args.end_frame, trajectory.shape[0])
    if start_frame >= end_frame:
        raise ValueError(f"empty frame range: start={start_frame}, end={end_frame}")
    trajectory = trajectory[start_frame:end_frame]

    robot = None
    overrun_count = 0
    phase_rows: list[str] = []
    frame_index_rows: list[int] = []
    elapsed_s_rows: list[float] = []
    target_pos_rad_rows: list[np.ndarray] = []
    actual_pos_rad_rows: list[np.ndarray] = []
    actual_vel_rad_s_rows: list[np.ndarray] = []
    actual_torque_rows: list[np.ndarray] = []
    error_rad_rows: list[np.ndarray] = []
    status_code_rows: list[np.ndarray] = []
    feedback_age_s_rows: list[np.ndarray] = []
    feedback_received_monotonic_s_rows: list[np.ndarray] = []
    feedback_received_wall_s_rows: list[np.ndarray] = []
    feedback_sequence_rows: list[np.ndarray] = []
    feedback_source_rows: list[np.ndarray] = []
    loop_time_s_rows: list[float] = []

    def append_tracking_sample(
        phase: str,
        frame_index: int,
        elapsed_s: float,
        target_rad: np.ndarray,
        states: dict,
        loop_time_s: float,
    ) -> np.ndarray:
        actual_pos, actual_vel, actual_torque, status_code = states_to_arrays(states, list(JOINT_NAMES))
        feedback_timing = states_to_timing_arrays(states, list(JOINT_NAMES))
        error_rad = np.asarray(target_rad, dtype=np.float64) - actual_pos
        phase_rows.append(phase)
        frame_index_rows.append(frame_index)
        elapsed_s_rows.append(elapsed_s)
        target_pos_rad_rows.append(np.asarray(target_rad, dtype=np.float64).copy())
        actual_pos_rad_rows.append(actual_pos)
        actual_vel_rad_s_rows.append(actual_vel)
        actual_torque_rows.append(actual_torque)
        error_rad_rows.append(error_rad)
        status_code_rows.append(status_code)
        feedback_age_s_rows.append(feedback_timing[0])
        feedback_received_monotonic_s_rows.append(feedback_timing[1])
        feedback_received_wall_s_rows.append(feedback_timing[2])
        feedback_sequence_rows.append(feedback_timing[3])
        feedback_source_rows.append(feedback_timing[4])
        loop_time_s_rows.append(loop_time_s)
        return error_rad

    def print_error_summary() -> None:
        if not error_rad_rows:
            return
        errors = np.vstack(error_rad_rows)
        replay_mask = np.asarray(phase_rows) == "replay"
        replay_errors = errors[replay_mask]
        print("Replay tracking error summary (target - actual):")
        for i, name in enumerate(JOINT_NAMES):
            values_deg = np.degrees(replay_errors[:, i])
            finite = values_deg[np.isfinite(values_deg)]
            if finite.size == 0:
                print(f"  {name:14s} no valid feedback")
                continue
            rmse = float(np.sqrt(np.mean(np.square(finite))))
            max_abs = float(np.max(np.abs(finite)))
            final_deg = float(np.degrees(errors[-1, i]))
            print(f"  {name:14s} rmse={rmse:7.3f} deg  max={max_abs:7.3f} deg  final={final_deg:+7.3f} deg")

    def save_tracking_log() -> None:
        if not args.log_out or not error_rad_rows:
            return
        out = Path(args.log_out).expanduser()
        torque_matrix = np.vstack(actual_torque_rows)
        gripper_index = list(JOINT_NAMES).index("gripper")
        save_npz(
            out,
            phase=np.asarray(phase_rows),
            frame_index=np.asarray(frame_index_rows, dtype=np.int64),
            elapsed_s=np.asarray(elapsed_s_rows, dtype=np.float64),
            joint_names=np.asarray(JOINT_NAMES),
            target_pos_rad=np.vstack(target_pos_rad_rows),
            actual_pos_rad=np.vstack(actual_pos_rad_rows),
            actual_vel_rad_s=np.vstack(actual_vel_rad_s_rows),
            actual_torque=torque_matrix,
            actual_torque_nm=torque_matrix,
            gripper_actual_torque_nm=torque_matrix[:, gripper_index],
            error_rad=np.vstack(error_rad_rows),
            status_code=np.vstack(status_code_rows),
            feedback_age_s=np.vstack(feedback_age_s_rows),
            feedback_received_monotonic_s=np.vstack(feedback_received_monotonic_s_rows),
            feedback_received_wall_s=np.vstack(feedback_received_wall_s_rows),
            feedback_sequence=np.vstack(feedback_sequence_rows),
            feedback_source=np.vstack(feedback_source_rows),
            loop_time_s=np.asarray(loop_time_s_rows, dtype=np.float64),
            source_file=np.asarray(str(Path(args.file).expanduser())),
            source_key=np.asarray(args.source),
            replay_fps=np.asarray(args.fps, dtype=np.int64),
            gripper_position_mit_kp=np.asarray(args.gripper_kp, dtype=np.float64),
            gripper_position_mit_kd=np.asarray(args.gripper_kd, dtype=np.float64),
            gripper_max_step_deg=np.asarray(args.gripper_max_step_deg, dtype=np.float64),
            gripper_torque_limit_0x700b_nm=np.asarray(args.gripper_torque_limit, dtype=np.float64),
        )
        print(f"Saved replay tracking log to {out}")

    try:
        robot_cfg = SeeedB601RSFollowerConfig(
            port=args.robot_port,
            can_adapter=args.can_adapter,
            id=args.robot_id,
            disable_torque_on_disconnect=not args.keep_torque_on_exit,
            gripper_control_mode="position_mit",
            gripper_position_mit_kp=args.gripper_kp,
            gripper_position_mit_kd=args.gripper_kd,
            gripper_max_step_deg=args.gripper_max_step_deg,
            gripper_torque_limit_0x700b_nm=args.gripper_torque_limit,
        )
        robot = SeeedB601RSFollower(robot_cfg)

        print(f"Connecting B601-RS on {args.robot_port}...")
        robot.connect(calibrate=args.calibrate_robot)
        print(
            "Gripper command: send_mit(target_pos, 0, "
            f"{args.gripper_kp:g}, {args.gripper_kd:g}, 0); "
            f"max_step={args.gripper_max_step_deg:g} deg/frame; "
            f"verified 0x700B={robot.active_gripper_torque_limit_nm:g} Nm"
        )

        joint_names = list(JOINT_NAMES)
        if joint_names != list(robot.motor_names):
            raise RuntimeError(f"unexpected robot joint order: {robot.motor_names}")

        gripper_limiter = GripperTorqueLimiter(
            kp=float(robot.config.gripper_mit_kp),
            kd=float(robot.config.gripper_mit_kd),
            torque_limit=float(robot.config.gripper_mit_torque_limit),
            hold_torque_limit=float(robot.config.gripper_mit_hold_torque_limit),
        )

        current_rad = current_positions_rad(robot, joint_names, limits_rad)
        zero_rad = np.zeros(len(joint_names), dtype=np.float64)
        first_rad = trajectory[0]

        print("Using conservative replay limits:")
        for name, (lo_deg, hi_deg) in CONSERVATIVE_LIMITS_DEG.items():
            print(f"  {name:14s} {lo_deg:+7.1f} .. {hi_deg:+7.1f} deg")

        prompt_before_motion(
            "The robot will move current RS pose -> zero pose -> first recorded frame -> replay trajectory.",
            assume_yes=args.yes,
        )
        move_smoothly(
            robot=robot,
            start_rad=current_rad,
            goal_rad=zero_rad,
            duration_s=args.zero_time,
            fps=args.fps,
            joint_names=joint_names,
            limits_rad=limits_rad,
            gripper_limiter=gripper_limiter,
            label="Move current RS pose to zero pose",
            gripper_max_step_rad=gripper_max_step_rad,
        )
        move_smoothly(
            robot=robot,
            start_rad=zero_rad,
            goal_rad=first_rad,
            duration_s=args.start_time,
            fps=args.fps,
            joint_names=joint_names,
            limits_rad=limits_rad,
            gripper_limiter=gripper_limiter,
            label="Move zero pose to first recorded frame",
            gripper_max_step_rad=gripper_max_step_rad,
        )

        print(f"Replaying {trajectory.shape[0]} frames from {args.source} at {args.fps} Hz.")
        start_s = time.perf_counter()
        for frame, target_rad in enumerate(trajectory):
            loop_start = time.perf_counter()
            states = request_and_read_states(robot, joint_names)
            send_rs_mit_targets(
                robot=robot,
                target_pos_rad=target_rad,
                joint_names=joint_names,
                states=states,
                gripper_limiter=gripper_limiter,
                dt_s=period_s,
            )
            loop_s = sleep_to_next_tick(loop_start, period_s)
            error_rad = append_tracking_sample(
                phase="replay",
                frame_index=start_frame + frame,
                elapsed_s=loop_start - start_s,
                target_rad=target_rad,
                states=states,
                loop_time_s=loop_s,
            )
            if loop_s > period_s * 1.15:
                overrun_count += 1
                if overrun_count <= 10 or overrun_count % args.fps == 0:
                    print(f"[warn] loop overrun frame={frame} loop={loop_s * 1000:.1f}ms target={period_s * 1000:.1f}ms")
            if frame and frame % args.fps == 0:
                elapsed = time.perf_counter() - start_s
                finite_error = np.abs(np.degrees(error_rad[np.isfinite(error_rad)]))
                max_error_deg = float(np.max(finite_error)) if finite_error.size else float("nan")
                print(
                    f"  frame={frame}/{trajectory.shape[0]} elapsed={elapsed:.1f}s "
                    f"max_error={max_error_deg:.2f} deg"
                )

        hold_steps = max(0, int(round(args.final_hold_time * args.fps)))
        if hold_steps:
            final_target_rad = trajectory[-1]
            print(f"Holding final target for {args.final_hold_time:.2f}s ({hold_steps} steps).")
            for hold_frame in range(hold_steps):
                loop_start = time.perf_counter()
                states = request_and_read_states(robot, joint_names)
                send_rs_mit_targets(
                    robot=robot,
                    target_pos_rad=final_target_rad,
                    joint_names=joint_names,
                    states=states,
                    gripper_limiter=gripper_limiter,
                    dt_s=period_s,
                )
                loop_s = sleep_to_next_tick(loop_start, period_s)
                append_tracking_sample(
                    phase="hold",
                    frame_index=start_frame + trajectory.shape[0] - 1,
                    elapsed_s=loop_start - start_s,
                    target_rad=final_target_rad,
                    states=states,
                    loop_time_s=loop_s,
                )

        print("Replay finished.")

    except KeyboardInterrupt:
        print("\nReplay stopped by Ctrl+C.")
    finally:
        print_error_summary()
        save_tracking_log()
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
            except Exception as exc:
                print(f"[warn] disconnect failed: {exc}")


if __name__ == "__main__":
    main()
