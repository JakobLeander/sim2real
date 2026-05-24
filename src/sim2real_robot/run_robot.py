
import math
import numpy as np
from sim2real_robot.robot_control_gpio import RobotControlGPIO
from sim2real_robot.robot_control_sim import RobotControlSim
from sim2real_robot.ai.onnx_infer import OnnxInfer
import time
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',filename='robot_runner.log', filemode='w')

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
        self._last_u = 0.0           # normalized last command [-1, 1], fed into obs
        self._last_motor_speed = 0.0 # actual rate-limited motor command [rad/s]
        self.fps = 0.0
        
        # drift in meters
        self._drift = 0.0

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
        # Called at 100hz
        pitch_angle = self.robot_control.get_pitch_angle()
        pitch_velocity = self.robot_control.get_pitch_velocity()
        velocity = self.robot_control.get_velocity()
        velocity=0
        self._drift += velocity * self.DT
        self._drift = 0

        obs = np.array([pitch_angle, pitch_velocity, self._last_u], dtype=np.float32)

        action = self._policy.infer(obs)

        # Scale raw policy output [-1, 1] to motor speed range
        max_velocity = 20.0
        raw_speed = max_velocity * float(action[0])

        # Rate-limit to match training (MAX_SPEED_CHANGE = 2.0 rad/s per step at 100 Hz)
        MAX_SPEED_CHANGE = 2.0
        delta = np.clip(raw_speed - self._last_motor_speed, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE)
        new_motor_speed = np.clip(self._last_motor_speed + delta, -max_velocity, max_velocity)

        # Store the actual command (normalized) as last_u for the next observation
        # This matches training: last_u_normalised = final_action[0] / 20.0
        self._last_motor_speed = new_motor_speed
        self._last_u = new_motor_speed / max_velocity

        self.robot_control.set_speed(new_motor_speed)

        self.write_logging(f"Pitch Angle: {pitch_angle:.2f} rad, Pitch Velocity: {pitch_velocity:.2f} rad/s,  Velocity: {velocity:.2f} m/s, Drift: {self._drift:.2f} m, Motor Speed: {new_motor_speed:.2f} rad/s, FPS: {self.fps:.2f}")

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

