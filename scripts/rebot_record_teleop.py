#!/usr/bin/env python3
"""Record B601-RS actual joint states while teleoperating from a reBot Arm 102 leader."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime

import numpy as np

from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from lerobot_robot_seeed_b601.seeed_b601_rs_follower import SeeedB601RSFollower
from lerobot_teleoperator_rebot_arm_102.config_rebot_arm_102_leader import RebotArm102LeaderConfig
from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader

from rebot_motion_common import (
    CONSERVATIVE_LIMITS_DEG,
    GripperTorqueLimiter,
    JOINT_NAMES,
    conservative_limits_rad,
    current_positions_rad,
    leader_action_to_rs_target_rad,
    limit_gripper_target_step,
    make_metadata,
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
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Leader arm serial port")
    parser.add_argument("--leader-baudrate", type=int, default=1_000_000)
    parser.add_argument("--robot-port", default="can0", help="B601-RS CAN channel")
    parser.add_argument("--can-adapter", default="socketcan", choices=["socketcan", "damiao"])
    parser.add_argument("--fps", type=int, default=150)
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds. Use <=0 for Ctrl+C stop.")
    parser.add_argument("--align-time", type=float, default=3.0, help="Smooth time from current RS pose to current leader pose")
    parser.add_argument("--gripper-kp", type=float, default=12.0)
    parser.add_argument("--gripper-kd", type=float, default=1.9)
    parser.add_argument("--gripper-max-step-deg", type=float, default=3.0)
    parser.add_argument("--gripper-torque-limit", type=float, default=0.5, help="Runtime 0x700B limit in Nm")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--leader-id", default="rebot_arm_102_leader")
    parser.add_argument("--robot-id", default="seeed_b601_rs_follower")
    parser.add_argument("--calibrate-leader", action="store_true", help="Allow leader calibration prompt during connect")
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

    teleop = None
    robot = None

    timestamps: list[float] = []
    leader_action_deg_rows: list[np.ndarray] = []
    rs_target_pos_rad_rows: list[np.ndarray] = []
    rs_actual_pos_rad_rows: list[np.ndarray] = []
    rs_actual_vel_rad_s_rows: list[np.ndarray] = []
    rs_actual_torque_rows: list[np.ndarray] = []
    rs_status_code_rows: list[np.ndarray] = []
    rs_feedback_age_s_rows: list[np.ndarray] = []
    rs_feedback_received_monotonic_s_rows: list[np.ndarray] = []
    rs_feedback_received_wall_s_rows: list[np.ndarray] = []
    rs_feedback_sequence_rows: list[np.ndarray] = []
    rs_feedback_source_rows: list[np.ndarray] = []
    loop_time_s_rows: list[float] = []
    overrun_count = 0
    skipped_feedback_frames = 0

    try:
        teleop_cfg = RebotArm102LeaderConfig(
            port=args.leader_port,
            baudrate=args.leader_baudrate,
            id=args.leader_id,
        )
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

        teleop = RebotArm102Leader(teleop_cfg)
        robot = SeeedB601RSFollower(robot_cfg)

        print(f"Connecting leader on {args.leader_port}...")
        teleop.connect(calibrate=args.calibrate_leader)
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
        leader_action = teleop.get_action()
        leader_goal_rad = leader_action_to_rs_target_rad(
            leader_action,
            joint_names,
            robot.config.joint_directions,
            limits_rad,
        )
        prompt_before_motion(
            "The robot will smoothly move from current RS pose to the current leader pose.",
            assume_yes=args.yes,
        )
        last_sent_target_rad = move_smoothly(
            robot=robot,
            start_rad=current_rad,
            goal_rad=leader_goal_rad,
            duration_s=args.align_time,
            fps=args.fps,
            joint_names=joint_names,
            limits_rad=limits_rad,
            gripper_limiter=gripper_limiter,
            label="Align RS to current leader pose",
            gripper_max_step_rad=gripper_max_step_rad,
        )

        print(f"Recording at {args.fps} Hz. Press Ctrl+C to stop.")
        start_s = time.perf_counter()
        frame = 0
        while args.duration <= 0 or (time.perf_counter() - start_s) < args.duration:
            loop_start = time.perf_counter()
            t_s = loop_start - start_s

            leader_action = teleop.get_action()
            leader_vec = np.array([float(leader_action[f"{name}.pos"]) for name in joint_names], dtype=np.float64)
            target_rad = leader_action_to_rs_target_rad(
                leader_action,
                joint_names,
                robot.config.joint_directions,
                limits_rad,
            )
            target_rad = limit_gripper_target_step(
                target_rad,
                last_sent_target_rad,
                joint_names,
                gripper_max_step_rad,
            )

            states = request_and_read_states(robot, joint_names)
            pos_rad, vel_rad_s, torque, status_code = states_to_arrays(states, joint_names)
            feedback_timing = states_to_timing_arrays(states, joint_names)
            send_rs_mit_targets(
                robot=robot,
                target_pos_rad=target_rad,
                joint_names=joint_names,
                states=states,
                gripper_limiter=gripper_limiter,
                dt_s=period_s,
            )
            last_sent_target_rad = target_rad

            saved_frame = bool(np.all(np.isfinite(pos_rad)))
            if not saved_frame:
                skipped_feedback_frames += 1
                if skipped_feedback_frames <= 10 or skipped_feedback_frames % args.fps == 0:
                    missing = [joint_names[i] for i, ok in enumerate(np.isfinite(pos_rad)) if not ok]
                    print(f"[warn] skipped frame={frame}, missing feedback: {missing}")

            loop_s = sleep_to_next_tick(loop_start, period_s)
            if saved_frame:
                timestamps.append(t_s)
                leader_action_deg_rows.append(leader_vec)
                rs_target_pos_rad_rows.append(target_rad)
                rs_actual_pos_rad_rows.append(pos_rad)
                rs_actual_vel_rad_s_rows.append(vel_rad_s)
                rs_actual_torque_rows.append(torque)
                rs_status_code_rows.append(status_code)
                rs_feedback_age_s_rows.append(feedback_timing[0])
                rs_feedback_received_monotonic_s_rows.append(feedback_timing[1])
                rs_feedback_received_wall_s_rows.append(feedback_timing[2])
                rs_feedback_sequence_rows.append(feedback_timing[3])
                rs_feedback_source_rows.append(feedback_timing[4])
                loop_time_s_rows.append(loop_s)
            if loop_s > period_s * 1.15:
                overrun_count += 1
                if overrun_count <= 10 or overrun_count % args.fps == 0:
                    print(f"[warn] loop overrun frame={frame} loop={loop_s * 1000:.1f}ms target={period_s * 1000:.1f}ms")
            frame += 1

    except KeyboardInterrupt:
        print("\nRecording stopped by Ctrl+C.")
    finally:
        if timestamps:
            torque_matrix = np.vstack(rs_actual_torque_rows)
            gripper_index = list(JOINT_NAMES).index("gripper")
            metadata = make_metadata(
                created_at=datetime.now().isoformat(timespec="seconds"),
                fps=args.fps,
                duration_s=args.duration,
                leader_port=args.leader_port,
                robot_port=args.robot_port,
                can_adapter=args.can_adapter,
                control_mode="robstride MIT direct motorbridge; gripper position_mit",
                main_replay_source="rs_target_pos_rad",
                gripper_position_mit_kp=(
                    float(robot.config.gripper_position_mit_kp) if robot is not None else args.gripper_kp
                ),
                gripper_position_mit_kd=(
                    float(robot.config.gripper_position_mit_kd) if robot is not None else args.gripper_kd
                ),
                gripper_max_step_deg=args.gripper_max_step_deg,
                gripper_torque_limit_0x700b_nm=(
                    robot.active_gripper_torque_limit_nm if robot is not None else args.gripper_torque_limit
                ),
                conservative_limits_deg=CONSERVATIVE_LIMITS_DEG,
                skipped_feedback_frames=skipped_feedback_frames,
                note=(
                    "RS actual state comes from timestamped type-0x18/type-0x02 CAN frames "
                    "before each target send; positions are radians and torque is N.m."
                ),
            )
            save_npz(
                args.out,
                timestamp_s=np.asarray(timestamps, dtype=np.float64),
                joint_names=np.asarray(JOINT_NAMES),
                limits_rad=limits_rad,
                limits_deg=np.asarray([CONSERVATIVE_LIMITS_DEG[name] for name in JOINT_NAMES], dtype=np.float64),
                leader_action_deg=np.vstack(leader_action_deg_rows),
                rs_target_pos_rad=np.vstack(rs_target_pos_rad_rows),
                rs_actual_pos_rad=np.vstack(rs_actual_pos_rad_rows),
                rs_actual_vel_rad_s=np.vstack(rs_actual_vel_rad_s_rows),
                rs_actual_torque=torque_matrix,
                rs_actual_torque_nm=torque_matrix,
                gripper_actual_torque_nm=torque_matrix[:, gripper_index],
                rs_status_code=np.vstack(rs_status_code_rows),
                rs_feedback_age_s=np.vstack(rs_feedback_age_s_rows),
                rs_feedback_received_monotonic_s=np.vstack(rs_feedback_received_monotonic_s_rows),
                rs_feedback_received_wall_s=np.vstack(rs_feedback_received_wall_s_rows),
                rs_feedback_sequence=np.vstack(rs_feedback_sequence_rows),
                rs_feedback_source=np.vstack(rs_feedback_source_rows),
                loop_time_s=np.asarray(loop_time_s_rows, dtype=np.float64),
                metadata_json=metadata,
            )
            print(f"Saved {len(timestamps)} frames to {args.out} (skipped_feedback_frames={skipped_feedback_frames})")

        for device in (teleop, robot):
            if device is not None:
                try:
                    if device.is_connected:
                        device.disconnect()
                except Exception as exc:
                    print(f"[warn] disconnect failed: {exc}")


if __name__ == "__main__":
    main()
