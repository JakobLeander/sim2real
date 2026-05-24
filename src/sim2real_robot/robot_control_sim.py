from sim2real_robot.robot_control_interfaces import RobotControl

class RobotControlSim(RobotControl):
    def __init__(self):
        super().__init__()

    def get_pitch_angle(self) -> float:
        return 0.0  # Placeholder for simulated pitch angle
    
    def start_robot(self):
        print("Starting robot in simulation...")

    def stop_robot(self):
        print("Stopping robot in simulation...")

    def set_speed(self, speed: float):
        print(f"Setting robot speed to {speed} rad/s in simulation...")

    def get_velocity(self):
        return 0.0  # Placeholder for simulated velocity

    def get_pitch_velocity(self) -> float:
        return 0.0  # Placeholder for simulated pitch velocity
