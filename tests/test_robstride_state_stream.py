import time
import unittest

import can

from lerobot_robot_seeed_b601.robstride_state_stream import (
    ROBSTRIDE_ACTIVE_REPORT,
    ROBSTRIDE_OPERATION_STATUS,
    RobStrideStateStream,
    TimestampedMotorState,
    decode_robstride_state_message,
)


def make_message(communication_type: int, motor_id: int = 7, host_id: int = 0xFD) -> can.Message:
    extra_data = (2 << 14) | (1 << 12) | motor_id
    arbitration_id = (communication_type << 24) | (extra_data << 8) | host_id
    return can.Message(
        timestamp=123.5,
        arbitration_id=arbitration_id,
        is_extended_id=True,
        data=b"\x7f\xff\x7f\xff\x7f\xff\x00\xfa",
    )


class RobStrideStateStreamTest(unittest.TestCase):
    def test_decode_state_message(self) -> None:
        cases = (
            (ROBSTRIDE_OPERATION_STATUS, "operation_status"),
            (ROBSTRIDE_ACTIVE_REPORT, "active_report"),
        )
        for communication_type, source in cases:
            with self.subTest(source=source):
                state = decode_robstride_state_message(
                    make_message(communication_type),
                    model="rs-00",
                    sequence=42,
                    received_monotonic_s=10.0,
                )

                self.assertIsNotNone(state)
                assert state is not None
                self.assertEqual(state.can_id, 7)
                self.assertEqual(state.source, source)
                self.assertEqual(state.sequence, 42)
                self.assertEqual(state.received_monotonic_s, 10.0)
                self.assertEqual(state.received_wall_s, 123.5)
                self.assertEqual(state.status_code, 1 << 4)
                self.assertAlmostEqual(state.pos, 0.0)
                self.assertAlmostEqual(state.vel, 0.0)
                self.assertAlmostEqual(state.torq, 0.0)
                self.assertAlmostEqual(state.t_mos, 25.0)

    def test_decode_uses_per_model_velocity_and_torque_limits(self) -> None:
        message = make_message(ROBSTRIDE_ACTIVE_REPORT)
        message.data = bytearray(b"\x7f\xff\xff\xfe\xff\xfe\x00\x00")

        state = decode_robstride_state_message(message, "rs-06", sequence=1)

        self.assertIsNotNone(state)
        assert state is not None
        self.assertAlmostEqual(state.vel, 20.0)
        self.assertAlmostEqual(state.torq, 36.0)

    def test_decode_ignores_unrelated_can_frames(self) -> None:
        message = make_message(0x11)
        self.assertIsNone(
            decode_robstride_state_message(message, "rs-00", sequence=1)
        )

    def test_snapshot_rejects_stale_feedback(self) -> None:
        stream = RobStrideStateStream("can0", {"joint": (1, 0xFD, "rs-00")})
        old_s = time.monotonic() - 1.0
        stream._states["joint"] = TimestampedMotorState(
            can_id=1,
            arbitration_id=0,
            status_code=0,
            pos=0.0,
            vel=0.0,
            torq=0.0,
            t_mos=25.0,
            t_rotor=0.0,
            received_monotonic_s=old_s,
            received_wall_s=time.time() - 1.0,
            sequence=1,
            source="active_report",
        )

        with self.assertRaisesRegex(TimeoutError, "stale=joint"):
            stream.snapshot(
                ["joint"],
                max_age_s=0.03,
                max_skew_s=0.02,
                wait_timeout_s=0.0,
            )

    def test_snapshot_reports_age_skew_and_sequences(self) -> None:
        stream = RobStrideStateStream(
            "can0",
            {
                "joint1": (1, 0xFD, "rs-06"),
                "joint2": (2, 0xFD, "rs-06"),
            },
        )
        now_s = time.monotonic()
        for index, name in enumerate(("joint1", "joint2"), start=1):
            stream._states[name] = TimestampedMotorState(
                can_id=index,
                arbitration_id=0,
                status_code=0,
                pos=0.1 * index,
                vel=0.0,
                torq=0.0,
                t_mos=25.0,
                t_rotor=0.0,
                received_monotonic_s=now_s - index * 0.001,
                received_wall_s=time.time(),
                sequence=10 + index,
                source="active_report",
            )

        states, diagnostics = stream.snapshot(
            ["joint1", "joint2"],
            max_age_s=0.03,
            max_skew_s=0.02,
            wait_timeout_s=0.0,
        )

        self.assertEqual(set(states), {"joint1", "joint2"})
        self.assertLess(diagnostics.max_age_s, 0.01)
        self.assertAlmostEqual(diagnostics.skew_s, 0.001)
        self.assertEqual(diagnostics.sequences, {"joint1": 11, "joint2": 12})


if __name__ == "__main__":
    unittest.main()
