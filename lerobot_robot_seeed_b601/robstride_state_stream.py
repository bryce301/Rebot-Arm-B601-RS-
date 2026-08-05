from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import can

logger = logging.getLogger(__name__)

ROBSTRIDE_OPERATION_STATUS = 0x02
ROBSTRIDE_ACTIVE_REPORT = 0x18
ROBSTRIDE_POSITION_MAX_RAD = 4.0 * math.pi

ROBSTRIDE_MODEL_LIMITS: dict[str, tuple[float, float, float]] = {
    "rs-00": (ROBSTRIDE_POSITION_MAX_RAD, 50.0, 14.0),
    "rs-01": (ROBSTRIDE_POSITION_MAX_RAD, 44.0, 17.0),
    "rs-02": (ROBSTRIDE_POSITION_MAX_RAD, 44.0, 17.0),
    "rs-03": (ROBSTRIDE_POSITION_MAX_RAD, 50.0, 60.0),
    "rs-04": (ROBSTRIDE_POSITION_MAX_RAD, 15.0, 120.0),
    "rs-05": (ROBSTRIDE_POSITION_MAX_RAD, 33.0, 17.0),
    "rs-06": (ROBSTRIDE_POSITION_MAX_RAD, 20.0, 36.0),
}


@dataclass(frozen=True)
class TimestampedMotorState:
    can_id: int
    arbitration_id: int
    status_code: int
    pos: float
    vel: float
    torq: float
    t_mos: float
    t_rotor: float
    received_monotonic_s: float
    received_wall_s: float
    sequence: int
    source: str


@dataclass(frozen=True)
class FeedbackDiagnostics:
    captured_monotonic_s: float
    max_age_s: float
    skew_s: float
    sequences: dict[str, int]


def _decode_signed_u16(raw: int, maximum: float) -> float:
    return (float(raw) / float(0x7FFF) - 1.0) * maximum


def decode_robstride_state_message(
    message: can.Message,
    model: str,
    sequence: int,
    received_monotonic_s: float | None = None,
) -> TimestampedMotorState | None:
    if not message.is_extended_id or message.dlc != 8:
        return None

    arbitration_id = int(message.arbitration_id)
    communication_type = (arbitration_id >> 24) & 0x1F
    if communication_type not in {ROBSTRIDE_OPERATION_STATUS, ROBSTRIDE_ACTIVE_REPORT}:
        return None

    limits = ROBSTRIDE_MODEL_LIMITS.get(model)
    if limits is None:
        raise ValueError(f"unsupported RobStride model for state decoding: {model}")
    p_max, v_max, t_max = limits

    extra_data = (arbitration_id >> 8) & 0xFFFF
    motor_id = extra_data & 0xFF
    payload = bytes(message.data)
    position_raw = int.from_bytes(payload[0:2], byteorder="big")
    velocity_raw = int.from_bytes(payload[2:4], byteorder="big")
    torque_raw = int.from_bytes(payload[4:6], byteorder="big")
    temperature_raw = int.from_bytes(payload[6:8], byteorder="big")

    status_code = 0
    status_code |= ((extra_data >> 13) & 0x01) << 5
    status_code |= ((extra_data >> 12) & 0x01) << 4
    status_code |= ((extra_data >> 11) & 0x01) << 3
    status_code |= ((extra_data >> 10) & 0x01) << 2
    status_code |= ((extra_data >> 9) & 0x01) << 1
    status_code |= (extra_data >> 8) & 0x01

    now_monotonic = time.monotonic() if received_monotonic_s is None else received_monotonic_s
    wall_timestamp = float(message.timestamp) if message.timestamp else time.time()
    source = "active_report" if communication_type == ROBSTRIDE_ACTIVE_REPORT else "operation_status"
    return TimestampedMotorState(
        can_id=motor_id,
        arbitration_id=arbitration_id,
        status_code=status_code,
        pos=_decode_signed_u16(position_raw, p_max),
        vel=_decode_signed_u16(velocity_raw, v_max),
        torq=_decode_signed_u16(torque_raw, t_max),
        t_mos=float(temperature_raw) * 0.1,
        t_rotor=0.0,
        received_monotonic_s=now_monotonic,
        received_wall_s=wall_timestamp,
        sequence=sequence,
        source=source,
    )


class RobStrideStateStream:
    """Timestamp RobStride status frames on a second SocketCAN receive socket."""

    def __init__(
        self,
        channel: str,
        motor_specs: dict[str, tuple[int, int, str]],
    ) -> None:
        self.channel = channel
        self._specs_by_id = {
            motor_id: (name, feedback_id, model)
            for name, (motor_id, feedback_id, model) in motor_specs.items()
        }
        self._states: dict[str, TimestampedMotorState] = {}
        self._sequences = {name: 0 for name in motor_specs}
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._bus: can.BusABC | None = None
        self._thread: threading.Thread | None = None
        self._receive_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        filters = [
            {
                "can_id": communication_type << 24,
                "can_mask": 0x1F000000,
                "extended": True,
            }
            for communication_type in (ROBSTRIDE_OPERATION_STATUS, ROBSTRIDE_ACTIVE_REPORT)
        ]
        self._bus = can.Bus(
            interface="socketcan",
            channel=self.channel,
            receive_own_messages=False,
            can_filters=filters,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"robstride-state-{self.channel}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.25)
        bus = self._bus
        if bus is not None:
            bus.shutdown()
        self._thread = None
        self._bus = None

    def _receive_loop(self) -> None:
        assert self._bus is not None
        while not self._stop_event.is_set():
            try:
                message = self._bus.recv(timeout=0.02)
            except Exception as exc:
                self._receive_error = exc
                logger.exception("RobStride state receiver stopped")
                with self._condition:
                    self._condition.notify_all()
                return
            if message is None:
                continue

            arbitration_id = int(message.arbitration_id)
            extra_data = (arbitration_id >> 8) & 0xFFFF
            motor_id = extra_data & 0xFF
            spec = self._specs_by_id.get(motor_id)
            if spec is None:
                continue
            name, _feedback_id, model = spec

            received_monotonic_s = time.monotonic()
            with self._condition:
                sequence = self._sequences[name] + 1
                state = decode_robstride_state_message(
                    message,
                    model,
                    sequence,
                    received_monotonic_s=received_monotonic_s,
                )
                if state is None:
                    continue
                self._sequences[name] = sequence
                self._states[name] = state
                self._condition.notify_all()

    def sequences(self) -> dict[str, int]:
        with self._condition:
            return dict(self._sequences)

    def wait_for_newer(self, baseline: dict[str, int], timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._receive_error is not None:
                    raise RuntimeError("RobStride state receiver failed") from self._receive_error
                missing = [
                    name
                    for name in self._sequences
                    if self._sequences[name] <= baseline.get(name, -1)
                ]
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "no new RobStride active report before timeout for: "
                        + ", ".join(missing)
                    )
                self._condition.wait(remaining)

    def snapshot(
        self,
        joint_names: list[str],
        max_age_s: float,
        max_skew_s: float,
        wait_timeout_s: float,
    ) -> tuple[dict[str, TimestampedMotorState], FeedbackDiagnostics]:
        deadline = time.monotonic() + wait_timeout_s
        with self._condition:
            while True:
                if self._receive_error is not None:
                    raise RuntimeError("RobStride state receiver failed") from self._receive_error
                captured_s = time.monotonic()
                missing = [name for name in joint_names if name not in self._states]
                states = {
                    name: self._states[name]
                    for name in joint_names
                    if name in self._states
                }
                stale = [
                    name
                    for name, state in states.items()
                    if captured_s - state.received_monotonic_s > max_age_s
                ]
                receipt_times = [state.received_monotonic_s for state in states.values()]
                skew_s = max(receipt_times) - min(receipt_times) if receipt_times else math.inf
                if not missing and not stale and skew_s <= max_skew_s:
                    max_age = max(captured_s - state.received_monotonic_s for state in states.values())
                    return states, FeedbackDiagnostics(
                        captured_monotonic_s=captured_s,
                        max_age_s=max_age,
                        skew_s=skew_s,
                        sequences={name: state.sequence for name, state in states.items()},
                    )

                remaining = deadline - captured_s
                if remaining <= 0.0:
                    details = []
                    if missing:
                        details.append("missing=" + ",".join(missing))
                    if stale:
                        details.append("stale=" + ",".join(stale))
                    if skew_s > max_skew_s:
                        details.append(f"skew={skew_s * 1000.0:.1f}ms")
                    raise TimeoutError("RobStride feedback is not fresh: " + " ".join(details))
                self._condition.wait(remaining)
