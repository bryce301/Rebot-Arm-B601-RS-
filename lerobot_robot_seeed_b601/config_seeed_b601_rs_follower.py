from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_rs_follower")
@dataclass
class SeeedB601RSFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    # Stream timestamped RobStride type-0x18 reports on a dedicated SocketCAN
    # receive socket. Firmware reports every 10 ms by default.
    use_active_report_state: bool = True
    state_max_age_s: float = 0.03
    state_max_skew_s: float = 0.02
    state_wait_timeout_s: float = 0.05

    # MIT control parameters for RS joints (ID1-ID6), gripper excluded.
    mit_kp: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 50.0,
            "shoulder_lift": 150.0,
            "elbow_flex": 150.0,
            "wrist_flex": 50.0,
            "wrist_yaw": 50.0,
            "wrist_roll": 50.0,
        }
    )

    mit_kd: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 3.0,
            "shoulder_lift": 10.0,
            "elbow_flex": 10.0,
            "wrist_flex": 5.0,
            "wrist_yaw": 4.0,
            "wrist_roll": 4.0,
        }
    )

    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (0x01, 0xFD),
            "shoulder_lift": (0x02, 0xFD),
            "elbow_flex":    (0x03, 0xFD),
            "wrist_flex":    (0x04, 0xFD),
            "wrist_yaw":     (0x05, 0xFD),
            "wrist_roll":    (0x06, 0xFD),
            "gripper":       (0x07, 0xFD),
        }
    )

    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (-145.0, 145.0),
            "shoulder_lift": (-0.0, 170.0),
            "elbow_flex":    (-0.0, 200.0),
            "wrist_flex":    (-80.0, 90.0),
            "wrist_yaw":     (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-0.0, 265.0),
        }
    )

    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 1.0,
            "shoulder_lift": 1.0,
            "elbow_flex": -1.0,
            "wrist_flex": -1.0,
            "wrist_yaw": -1.0,
            "wrist_roll": 1.0,
            "gripper": 6.0,
        }
    )

    # The Kp parameter for the MIT control mode of the gripper.
    gripper_mit_kp: float = 12.0

    # The Kd parameter for the MIT control mode of the gripper.
    gripper_mit_kd: float = 0.05

    # Max absolute output torque allowed in MIT mode for the gripper
    gripper_mit_torque_limit: float = 3.5

    # Max absolute output torque for gripper when target velocity is near zero
    # (used as grasp/hold state torque limit).
    gripper_mit_hold_torque_limit: float = 1.0

    # Gripper controller selected by the LeRobot CLI:
    # "torque" keeps the existing software torque controller;
    # "position_mit" sends the target angle through motor-side MIT position control.
    gripper_control_mode: str = "torque"

    # Motor-side MIT position gains used when gripper_control_mode="position_mit".
    gripper_position_mit_kp: float = 12.0
    gripper_position_mit_kd: float = 1.9

    # RobStride total torque limit written to 0x700B for position_mit mode.
    # The value remains active after disconnect.
    gripper_torque_limit_0x700b_nm: float = 0.5
    gripper_torque_limit_verify_tolerance_nm: float = 0.02


    # The v_des parameter for the position-velocity control mode of the joints.
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [50, 0.4, 0.4, 50, 50, 50, 0]
    )
