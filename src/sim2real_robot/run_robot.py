
import math
import numpy as np
from sim2real_robot.robot_control_gpio import RobotControlGPIO
from sim2real_robot.robot_control_sim import RobotControlSim
from sim2real_robot.ai.onnx_infer import OnnxInfer

import time



class RobotRunner:
    CONTROL_RATE_HZ = 100
    DT = 1/CONTROL_RATE_HZ

    def __init__(self, backend: str = "gpio"):
        if backend == "real":
            self.robot_control = RobotControlGPIO()
        elif backend == "sim":
            self.robot_control = RobotControlSim()
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported backends: 'gpio', 'sim'")
        
        self._policy = OnnxInfer()
        
        self._fps_count = 0
        self._fps_time_start = time.time()
        self._fps_elapsed = 0.0
        self.fps=0.0
        
        # drift in meters
        self._drift = 0.0

    def write_debug(self, text: str):
        print(text, flush=True, end="\r")

    def calculate_fps(self):
        self._fps_count += 1
        self._fps_elapsed = time.time() - self._fps_time_start
        if self._fps_elapsed >= 1.0:
            self.fps = self._fps_count / self._fps_elapsed
            self._fps_count = 0
            self._fps_time_start = time.time()
            self.write_debug(f"FPS: {self.fps:.2f}")

    def update(self):
        # Fetch data for our observation
        pitch_angle = self.robot_control.get_pitch_angle()
        pitch_velocity = self.robot_control.get_pitch_velocity()
        velocity = self.robot_control.get_velocity()
        self._drift += velocity * self.DT

        obs = np.array([pitch_angle, pitch_velocity, self._drift, velocity], dtype=np.float32)

        action = self._policy.infer(obs)
        
        # Since we normalized the action to be between -1 and 1 during training, we need to scale it back to the actual speed range of the robot
        max_velocity = 5.0
        new_motor_speed = max_velocity * float(action[0])  # Scale action to motor speed range
        print(new_motor_speed)
        self.robot_control.set_speed(new_motor_speed)

    def run(self):
        fps_count = 0
        fps_time_start = time.time()
        fps_elapsed = 0.0
        next_time = time.perf_counter()

        try:
            self.robot_control.start_robot()

            while True:
                if time.perf_counter()>=next_time:
                    self.update()  # Update robot state
                    self.calculate_fps()

                    next_time += self.DT
                    time.sleep(0.001)            

        finally:
            self.robot_control.stop_robot()
            print("\nRobot stopped. Exiting.")

if __name__ == "__main__":
    runner = RobotRunner(backend="real")  # Change to "sim" for simulation or "gpio" for real hardware
    runner.run()

