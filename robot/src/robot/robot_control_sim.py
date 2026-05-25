import numpy as np
from robot.robot_control_interfaces import RobotControl
from shared.constants import RobotSpec


class RobotControlSim(RobotControl):
    """Simulated robot that returns sensor-like values with Gaussian noise.

    The pitch angle drifts slightly with the applied motor command so the
    runner's policy receives plausible, correlated observations rather than
    pure white noise.
    """

    # Noise standard deviations (tuned to match typical MPU-6050 noise floor)
    _PITCH_ANGLE_NOISE_STD = 0.01  # rad
    _PITCH_VEL_NOISE_STD = 0.05  # rad/s
    _VELOCITY_NOISE_STD = 0.005  # m/s
    _WHEEL_RADIUS_M = RobotSpec.WHEEL_RADIUS

    def __init__(self):
        super().__init__()
        self._pitch_angle: float = 0.0  # rad, ~0 = upright
        self._motor_speed: float = 0.0  # rad/s, last command from set_speed

    def start_robot(self):
        print("Starting robot in simulation...")

    def stop_robot(self):
        print("Stopping robot in simulation...")

    def set_speed(self, speed: float):
        self._motor_speed = speed

    def get_pitch_angle(self) -> float:
        """Returns a noisy pitch angle near upright (0 rad)."""
        noise = np.random.normal(0.0, self._PITCH_ANGLE_NOISE_STD)
        return self._pitch_angle + noise

    def get_pitch_velocity(self) -> float:
        """Returns noisy pitch angular velocity in rad/s."""
        return float(np.random.normal(0.0, self._PITCH_VEL_NOISE_STD))

    def get_velocity(self) -> float:
        """Returns wheel velocity converted to m/s with sensor noise."""
        ideal = self._motor_speed * self._WHEEL_RADIUS_M
        noise = np.random.normal(0.0, self._VELOCITY_NOISE_STD)
        return float(ideal + noise)
