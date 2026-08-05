#!/usr/bin/env python3
"""Teleoperate B601-RS with direct MIT gripper position control and 0x700B limiting."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import (
    SeeedB601RSFollowerConfig,
)
from lerobot_robot_seeed_b601.seeed_b601_rs_follower import (
    SeeedB601RSFollower,
)
from lerobot_teleoperator_rebot_arm_102.config_rebot_arm_102_leader import (
    RebotArm102LeaderConfig,
)
from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader


CSV_FIELDS = [
    "frame",
    "phase",
    "elapsed_s",
    "actual_call_dt_s",
    "leader_gripper_deg",
    "commanded_gripper_deg",
    "commanded_gripper_rad",
    "current_gripper_deg",
    "current_gripper_rad",
    "position_error_deg",
    "position_error_rad",
    "state_velocity_rad_s",
    "torque_feedback_nm",
    "mos_temperature_c",
    "status_code",
    "feedback_age_ms",
    "feedback_sequence",
    "feedback_source",
    "mit_kp",
    "mit_kd",
    "tau_ff_nm",
    "torque_limit_0x700b_nm",
]


def minimum_jerk(progress: float) -> float:
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def sleep_to_period(loop_start_s: float, period_s: float) -> None:
    remaining_s = period_s - (time.perf_counter() - loop_start_s)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def read_current_positions_deg(
    robot: SeeedB601RSFollower,
    attempts: int = 10,
) -> dict[str, float]:
    for attempt in range(1, attempts + 1):
        try:
            states = robot.get_motor_states()
        except Exception:
            states = {}

        positions = {}
        for name, state in states.items():
            if state is not None:
                positions[name] = math.degrees(float(state.pos))
        if len(positions) == len(robot.motors):
            return positions
        if attempt < attempts:
            time.sleep(0.02)

    missing = sorted(set(robot.motors) - set(positions))
    raise RuntimeError(f"missing RS feedback for joints: {missing}")


class TeleopCsvLogger:
    def __init__(
        self,
        robot: SeeedB601RSFollower,
        writer: csv.DictWriter,
        log_file: Any,
    ) -> None:
        self.robot = robot
        self.writer = writer
        self.log_file = log_file
        self.start_s = time.perf_counter()
        self.previous_call_s: float | None = None
        self.frame = 0

    def send_and_log(self, action: dict[str, float], phase: str) -> None:
        call_s = time.perf_counter()
        call_dt_s = (
            math.nan
            if self.previous_call_s is None
            else call_s - self.previous_call_s
        )
        self.previous_call_s = call_s

        sent_action = self.robot.send_action(action)
        commanded_deg = float(sent_action["gripper.pos"])
        leader_gripper_deg = float(action["gripper.pos"])
        state = self.robot.get_motor_states().get("gripper")

        if state is None:
            current_rad = math.nan
            state_vel = math.nan
            torque = math.nan
            temperature = math.nan
            status = -1
        else:
            current_rad = float(state.pos)
            state_vel = float(state.vel)
            torque = float(state.torq)
            temperature = float(state.t_mos)
            status = int(state.status_code)

        received_s = getattr(state, "received_monotonic_s", None)
        feedback_age_ms = (
            math.nan
            if received_s is None
            else max(0.0, time.monotonic() - float(received_s)) * 1000.0
        )

        commanded_rad = math.radians(commanded_deg)
        self.writer.writerow(
            {
                "frame": self.frame,
                "phase": phase,
                "elapsed_s": call_s - self.start_s,
                "actual_call_dt_s": call_dt_s,
                "leader_gripper_deg": leader_gripper_deg,
                "commanded_gripper_deg": commanded_deg,
                "commanded_gripper_rad": commanded_rad,
                "current_gripper_deg": math.degrees(current_rad),
                "current_gripper_rad": current_rad,
                "position_error_deg": commanded_deg - math.degrees(current_rad),
                "position_error_rad": commanded_rad - current_rad,
                "state_velocity_rad_s": state_vel,
                "torque_feedback_nm": torque,
                "mos_temperature_c": temperature,
                "status_code": status,
                "feedback_age_ms": feedback_age_ms,
                "feedback_sequence": getattr(state, "sequence", -1),
                "feedback_source": getattr(state, "source", "unknown"),
                "mit_kp": self.robot.config.gripper_position_mit_kp,
                "mit_kd": self.robot.config.gripper_position_mit_kd,
                "tau_ff_nm": 0.0,
                "torque_limit_0x700b_nm": self.robot.active_gripper_torque_limit_nm,
            }
        )
        self.frame += 1
        if self.frame % 150 == 0:
            self.log_file.flush()


def smooth_to_leader_pose(
    robot: SeeedB601RSFollower,
    leader_action: dict[str, float],
    csv_logger: TeleopCsvLogger,
    fps: float,
    minimum_duration_s: float,
    max_joint_speed_deg_s: float,
) -> None:
    current_deg = read_current_positions_deg(robot)
    initial_action: dict[str, float] = {}
    for name in robot.motor_names:
        key = f"{name}.pos"
        if key not in leader_action:
            raise KeyError(f"leader action missing {key}")
        direction = float(robot.config.joint_directions[name])
        if direction == 0.0:
            raise ValueError(f"joint direction for {name} must be non-zero")
        initial_action[key] = current_deg[name] / direction

    target_follower_deg: dict[str, float] = {}
    for name in robot.motor_names:
        key = f"{name}.pos"
        position = float(leader_action[key]) * float(robot.config.joint_directions[name])
        if name in robot.config.joint_limits:
            lower_deg, upper_deg = robot.config.joint_limits[name]
            position = max(lower_deg, min(upper_deg, position))
        target_follower_deg[name] = position
    max_delta_deg = max(
        abs(target_follower_deg[name] - current_deg[name])
        for name in robot.motor_names
    )
    # A minimum-jerk curve has a peak normalized slope of 1.875.
    duration_s = max(
        minimum_duration_s,
        1.875 * max_delta_deg / max_joint_speed_deg_s,
    )
    steps = max(2, math.ceil(duration_s * fps) + 1)
    period_s = 1.0 / fps
    print(
        f"Startup transition: {duration_s:.2f}s, {steps} frames, "
        f"largest joint move={max_delta_deg:.2f} deg"
    )

    for step in range(steps):
        loop_start_s = time.perf_counter()
        progress = step / (steps - 1)
        blend = minimum_jerk(progress)
        action = {
            key: initial_action[key]
            + blend * (float(leader_action[key]) - initial_action[key])
            for key in initial_action
        }
        csv_logger.send_and_log(action, "startup")
        sleep_to_period(loop_start_s, period_s)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default="/dev/ttyUSB0")
    parser.add_argument("--leader-baudrate", type=int, default=1_000_000)
    parser.add_argument("--leader-id", default="rebot_arm_102_leader")
    parser.add_argument("--robot-port", default="can0")
    parser.add_argument("--robot-id", default="follower1")
    parser.add_argument("--can-adapter", default="socketcan", choices=["socketcan", "damiao"])
    parser.add_argument("--fps", type=float, default=150.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; <=0 runs until Ctrl+C")
    parser.add_argument("--startup-min-duration", type=float, default=3.0)
    parser.add_argument("--startup-max-speed", type=float, default=30.0, help="Maximum follower joint speed in deg/s")
    parser.add_argument("--gripper-kp", type=float, default=12.0)
    parser.add_argument("--gripper-kd", type=float, default=2.0)
    parser.add_argument("--gripper-torque-limit", type=float, default=0.5, help="Runtime 0x700B limit in Nm")
    parser.add_argument(
        "--out",
        default=f"logs/gripper_position_mit_{timestamp}.csv",
    )
    parser.add_argument("--calibrate-leader", action="store_true")
    parser.add_argument("--calibrate-robot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.startup_min_duration < 0.0:
        raise ValueError("--startup-min-duration must be >= 0")
    if args.startup_max_speed <= 0.0:
        raise ValueError("--startup-max-speed must be positive")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.fps

    leader = None
    robot = None
    csv_logger = None
    with out_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            leader = RebotArm102Leader(
                RebotArm102LeaderConfig(
                    port=args.leader_port,
                    baudrate=args.leader_baudrate,
                    id=args.leader_id,
                )
            )
            robot = SeeedB601RSFollower(
                SeeedB601RSFollowerConfig(
                    port=args.robot_port,
                    can_adapter=args.can_adapter,
                    id=args.robot_id,
                    disable_torque_on_disconnect=True,
                    gripper_control_mode="position_mit",
                    gripper_position_mit_kp=args.gripper_kp,
                    gripper_position_mit_kd=args.gripper_kd,
                    gripper_torque_limit_0x700b_nm=args.gripper_torque_limit,
                )
            )

            print(f"Connecting leader on {args.leader_port}...")
            leader.connect(calibrate=args.calibrate_leader)
            print(f"Connecting B601-RS on {args.robot_port}...")
            robot.connect(calibrate=args.calibrate_robot)
            csv_logger = TeleopCsvLogger(robot, writer, log_file)

            print(
                "Gripper command: send_mit(target_pos, 0, "
                f"{args.gripper_kp:g}, {args.gripper_kd:g}, 0); "
                f"verified 0x700B={robot.active_gripper_torque_limit_nm:g} Nm"
            )
            print("Keep the leader still during the startup transition.")
            leader_pose = leader.get_action()
            smooth_to_leader_pose(
                robot,
                leader_pose,
                csv_logger,
                args.fps,
                args.startup_min_duration,
                args.startup_max_speed,
            )
            print(f"Teleoperating at {args.fps:g} Hz. Press Ctrl+C to stop.")
            print(f"Logging to {out_path}")

            start_s = time.perf_counter()
            while args.duration <= 0.0 or time.perf_counter() - start_s < args.duration:
                loop_start_s = time.perf_counter()
                csv_logger.send_and_log(leader.get_action(), "teleop")
                sleep_to_period(loop_start_s, period_s)

        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
        finally:
            log_file.flush()
            try:
                if leader is not None and leader.is_connected:
                    leader.disconnect()
            finally:
                if robot is not None and robot.is_connected:
                    robot.disconnect()

    frames = 0 if csv_logger is None else csv_logger.frame
    print(f"Saved {frames} gripper frames to {out_path}")


if __name__ == "__main__":
    main()
