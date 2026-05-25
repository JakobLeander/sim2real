from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    # Physical parameters
    CONTROL_DT: float = 0.01  # control timestep (s)
    MAX_VELOCITY: float = 20.0  # max motor velocity (rad/s)
    MAX_DELTA_VEL: float = 2.0  # max rad/s change per control step
    WHEEL_RADIUS: float = 0.053  # wheel radius (m)
