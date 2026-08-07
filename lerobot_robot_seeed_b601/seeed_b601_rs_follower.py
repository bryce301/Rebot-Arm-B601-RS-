import logging
import math
from typing import Any

from .config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from .robstride_state_stream import FeedbackDiagnostics, RobStrideStateStream
from .seeed_b601_follower import FOLLOWER_GRIPPER_MOTOR, SeeedB601FollowerBase

logger = logging.getLogger(__name__)

GRIPPER_CONTROL_MODES = {"torque", "position_mit"}
GRIPPER_TORQUE_LIMIT_PARAM_ID = 0x700B
ROBSTRIDE_RS00_MAX_TORQUE_NM = 14.0


class SeeedB601RSFollower(SeeedB601FollowerBase):
    """
    Seeed B601-RS Robot Arm (RobStride Motors).
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601RSFollowerConfig
    name = "seeed_b601_rs_follower"
    motor_type = "rs"

    motor_model_mapping = {
        "shoulder_pan":  "rs-06",
        "shoulder_lift": "rs-06",
        "elbow_flex":    "rs-06",
        "wrist_flex":    "rs-00",
        "wrist_yaw":     "rs-00",
        "wrist_roll":    "rs-00",
        "gripper":       "rs-00",
    }

    def __init__(self, config: SeeedB601RSFollowerConfig):
        super().__init__(config)
        self._active_gripper_torque_limit_nm: float | None = None
        self._state_stream: RobStrideStateStream | None = None
        self._last_feedback_diagnostics: FeedbackDiagnostics | None = None

    @property
    def active_gripper_torque_limit_nm(self) -> float | None:
        return self._active_gripper_torque_limit_nm

    @property
    def last_feedback_diagnostics(self) -> FeedbackDiagnostics | None:
        return self._last_feedback_diagnostics

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.motor_model_mapping[motor_name]
            self.motors[motor_name] = self.bus.add_robstride_motor(send_id, recv_id, motor_type_str)

    def _configure_before_enable(self) -> None:
        gripper_control_mode = str(self.config.gripper_control_mode)
        if gripper_control_mode not in GRIPPER_CONTROL_MODES:
            choices = ", ".join(sorted(GRIPPER_CONTROL_MODES))
            raise ValueError(
                f"unsupported gripper_control_mode={gripper_control_mode!r}; "
                f"expected one of: {choices}"
            )
        if float(self.config.gripper_max_step_deg) < 0.0:
            raise ValueError("gripper_max_step_deg must be >= 0")
        if gripper_control_mode == "torque":
            self._active_gripper_torque_limit_nm = None
            logger.info("Gripper control mode: torque")
        else:
            requested_limit = float(self.config.gripper_torque_limit_0x700b_nm)
            if not 0.0 < requested_limit <= ROBSTRIDE_RS00_MAX_TORQUE_NM:
                raise ValueError(
                    "gripper_torque_limit_0x700b_nm must be in "
                    f"(0, {ROBSTRIDE_RS00_MAX_TORQUE_NM:g}]"
                )

            gripper = self.motors.get(FOLLOWER_GRIPPER_MOTOR)
            if gripper is None:
                raise RuntimeError("gripper motor is not available")

            gripper.robstride_write_param_f32(
                GRIPPER_TORQUE_LIMIT_PARAM_ID, requested_limit
            )
            actual_limit = float(
                gripper.robstride_get_param_f32(GRIPPER_TORQUE_LIMIT_PARAM_ID, 1000)
            )
            tolerance = float(self.config.gripper_torque_limit_verify_tolerance_nm)
            if not math.isclose(
                actual_limit, requested_limit, rel_tol=0.0, abs_tol=tolerance
            ):
                raise RuntimeError(
                    "gripper 0x700B verification failed: "
                    f"requested={requested_limit:.3f} Nm, read={actual_limit:.3f} Nm"
                )
            self._active_gripper_torque_limit_nm = actual_limit
            logger.info(
                "Gripper control mode: position_mit (kp=%.3f, kd=%.3f, 0x700B=%.3f Nm)",
                float(self.config.gripper_position_mit_kp),
                float(self.config.gripper_position_mit_kd),
                actual_limit,
            )

        if self.config.use_active_report_state:
            if self.config.can_adapter != "socketcan":
                raise ValueError("timestamped RobStride state requires can_adapter='socketcan'")
            motor_specs = {
                name: (
                    self.config.motor_can_ids[name][0],
                    self.config.motor_can_ids[name][1],
                    self.motor_model_mapping[name],
                )
                for name in self.motor_names
            }
            stream = RobStrideStateStream(self.config.port, motor_specs)
            stream.start()
            self._state_stream = stream
            try:
                for motor in self.motors.values():
                    motor.robstride_set_active_report(True)
            except Exception:
                self._stop_state_stream()
                raise
            logger.info("RobStride 10 ms active-report state stream enabled")

    def _configure_after_enable(self) -> None:
        if self._state_stream is None:
            return
        baseline = self._state_stream.sequences()
        try:
            self._state_stream.wait_for_newer(baseline, timeout_s=0.25)
            self.get_motor_states()
        except Exception:
            self._stop_state_stream()
            raise
        diag = self._last_feedback_diagnostics
        if diag is not None:
            logger.info(
                "RobStride feedback ready: max_age=%.2fms skew=%.2fms",
                diag.max_age_s * 1000.0,
                diag.skew_s * 1000.0,
            )

    def get_motor_states(self) -> dict[str, Any | None]:
        if self._state_stream is None:
            return super().get_motor_states()
        states, diagnostics = self._state_stream.snapshot(
            list(self.motor_names),
            max_age_s=float(self.config.state_max_age_s),
            max_skew_s=float(self.config.state_max_skew_s),
            wait_timeout_s=float(self.config.state_wait_timeout_s),
        )
        self._last_feedback_diagnostics = diagnostics
        return states

    def _stop_state_stream(self) -> None:
        if self._state_stream is None:
            return
        for motor in self.motors.values():
            try:
                motor.robstride_set_active_report(False)
            except Exception:
                logger.warning("Failed to disable RobStride active report", exc_info=True)
        self._state_stream.stop()
        self._state_stream = None

    def _configure_before_disconnect(self) -> None:
        self._stop_state_stream()

    def mit_output_torque_limit(
        self,
        motor,
        pos_target: float,
    ) -> float | None:
        if motor is None:
            return None
        state = self.get_motor_states().get(FOLLOWER_GRIPPER_MOTOR)
        if state is None:
            return None

        # Impedance control: tau = K*(x_r - x) + B*(x_dot_r - x_dot)
        control_dt_s = 1.0 / 150.0
        # Assumed control frequency.
        if control_dt_s <= 0.0:
            return None

        if not hasattr(self, "_gripper_prev_target_pos"):
            self._gripper_prev_target_pos = None
        if not hasattr(self, "_gripper_prev_filtered_target_vel"):
            self._gripper_prev_filtered_target_vel = None
        if not hasattr(self, "_gripper_prev_state_pos"):
            self._gripper_prev_state_pos = None

        prev_target_pos = self._gripper_prev_target_pos
        if prev_target_pos is None:
            target_vel = 0.0
        else:
            target_vel = (pos_target - prev_target_pos) / control_dt_s
        self._gripper_prev_target_pos = pos_target

        lpf_alpha = 0.3
        target_vel_max = 3.0
        prev_filtered_vel = self._gripper_prev_filtered_target_vel
        if prev_filtered_vel is None:
            filtered_target_vel = target_vel
        else:
            filtered_target_vel = (
                lpf_alpha * target_vel + (1.0 - lpf_alpha) * prev_filtered_vel
            )
        target_vel = max(-target_vel_max, min(target_vel_max, filtered_target_vel))
        self._gripper_prev_filtered_target_vel = target_vel

        prev_state_pos = self._gripper_prev_state_pos
        if prev_state_pos is None:
            estimated_state_vel = 0.0
        else:
            estimated_state_vel = (state.pos - prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos

        kp = float(self.config.gripper_mit_kp)
        kd = float(self.config.gripper_mit_kd)
        impedance_torque = (
            kp * (pos_target - state.pos)
            + kd * (target_vel - state.vel)
        )
        logger.debug(
            "Gripper MIT terms: pos_target=%.4f rad, state_pos=%.4f rad, "
            "target_vel=%.4f rad/s, state_vel=%.4f rad/s",
            pos_target,
            state.pos,
            target_vel,
            state.vel,
        )

        # Use dedicated grasp/hold torque limit when estimated motor speed is small.
        max_torque = (
            self.config.gripper_mit_torque_limit
            if abs(estimated_state_vel) > 0.25
            else self.config.gripper_mit_hold_torque_limit
        )
        motor.request_feedback()
        return max(-max_torque, min(max_torque, impedance_torque))
