from pathlib import Path
import numpy as np

from typing import TYPE_CHECKING

from robot.robot_control_sim import RobotControlSim

if TYPE_CHECKING:
    from robot.robot_control_gpio import RobotControlGPIO
from robot.ai.onnx_infer import OnnxInfer
import time
import logging

from shared.constants import RobotSpec

_LOG_DIR = Path(__file__).parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    filename=_LOG_DIR / "robot_runner.log",
    filemode="w",
)


class RobotRunner:
    def __init__(self, backend: str, onnx_policy_path: str):
        if backend == "real":
            from robot.robot_control_gpio import RobotControlGPIO
            self.robot_control = RobotControlGPIO()
        elif backend == "sim":
            self.robot_control = RobotControlSim()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported backends: 'gpio', 'sim'"
            )

        self._policy = OnnxInfer(onnx_policy_path)

        self._fps_count = 0
        self._fps_time_start = time.time()
        self._fps_elapsed = 0.0
        self._last_u = 0.0  # normalized last command [-1, 1], fed into obs
        self._last_motor_speed = 0.0  # actual rate-limited motor command [rad/s]
        self.fps = 0.0

    def write_logging(self, text: str):
        logging.info(text)

    def calculate_fps(self):
        self._fps_count += 1
        self._fps_elapsed = time.time() - self._fps_time_start
        if self._fps_elapsed >= 1.0:
            self.fps = self._fps_count / self._fps_elapsed
            self._fps_count = 0
            self._fps_time_start = time.time()

    def update(self):
        pitch_angle = self.robot_control.get_pitch_angle()
        pitch_velocity = self.robot_control.get_pitch_velocity()
        velocity = self.robot_control.get_velocity()
        velocity = 0

        obs = np.array([pitch_angle, pitch_velocity, self._last_u], dtype=np.float32)

        action = self._policy.infer(obs)

        # Scale raw policy output [-1, 1] to motor speed range
        MAX_VELOCITY = RobotSpec.MAX_VELOCITY
        raw_speed = MAX_VELOCITY * float(action[0])

        # Rate-limit to match training (MAX_SPEED_CHANGE = 2.0 rad/s per step at 100 Hz)
        MAX_DELTA_VEL = RobotSpec.MAX_DELTA_VEL
        delta = np.clip(
            raw_speed - self._last_motor_speed, -MAX_DELTA_VEL, MAX_DELTA_VEL
        )
        new_motor_speed = np.clip(
            self._last_motor_speed + delta, -MAX_VELOCITY, MAX_VELOCITY
        )

        # Store the actual command (normalized) as last_u for the next observation
        # This matches training: last_u_normalised = final_action[0] / 20.0
        self._last_motor_speed = new_motor_speed
        self._last_u = new_motor_speed / MAX_VELOCITY

        self.robot_control.set_speed(new_motor_speed)

        self.write_logging(
            f"Pitch Angle: {pitch_angle:.2f} rad, Pitch Velocity: {pitch_velocity:.2f} rad/s,  Velocity: {velocity:.2f} m/s, Motor Speed: {new_motor_speed:.2f} rad/s, FPS: {self.fps:.2f}"
        )

    def run(self):
        fps_count = 0
        fps_time_start = time.time()
        fps_elapsed = 0.0
        CONTROL_DT = RobotSpec.CONTROL_DT

        next_time = time.perf_counter()

        try:
            self.robot_control.start_robot()

            while True:
                if time.perf_counter() >= next_time:
                    self.update()  # Update robot state
                    self.calculate_fps()

                    next_time += CONTROL_DT
                    time.sleep(0.001)

        finally:
            self.robot_control.stop_robot()
            print("\nRobot stopped. Exiting.")


if __name__ == "__main__":
    backend = "real"  # Change to "sim" for simulation or "real" for real hardware
    onnx_policy_path = Path(__file__).parents[3] / "policies" / "robot_policy.onnx"

    runner = RobotRunner(backend=backend, onnx_policy_path=str(onnx_policy_path))
    runner.run()
