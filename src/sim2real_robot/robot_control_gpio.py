from .robot_control_interfaces import RobotControl
from .hardware.imu import IMU
from .hardware.motor import StepperMotor


class RobotControlGPIO(RobotControl):
    MOTOR_LEFT_ENABLE_PIN = 21
    MOTOR_LEFT_STEP_PIN = 12
    MOTOR_LEFT_DIR_PIN = 20

    MOTOR_RIGHT_ENABLE_PIN = 26
    MOTOR_RIGHT_STEP_PIN = 19
    MOTOR_RIGHT_DIR_PIN = 13
    def __init__(self):
        super().__init__()

        self._imu = IMU()  # Initialize the IMU sensor
        self._left_motor = StepperMotor(step_pin=self.MOTOR_LEFT_STEP_PIN, dir_pin=self.MOTOR_LEFT_DIR_PIN, enable_pin=self.MOTOR_LEFT_ENABLE_PIN, reverse_direction=False)  # Initialize left motor
        self._right_motor = StepperMotor(step_pin=self.MOTOR_RIGHT_STEP_PIN, dir_pin=self.MOTOR_RIGHT_DIR_PIN, enable_pin=self.MOTOR_RIGHT_ENABLE_PIN, reverse_direction=False)  # Initialize right motor
        self._imu = IMU()  # Initialize the IMU sensor

    def get_pitch_angle(self) -> float:
        # Implement GPIO-based method to read pitch angle
        # Placeholder implementation
        pitch_angle = self._imu.pitch_angle  # Get pitch angle from IMU
        return pitch_angle

    def start_robot(self):
        # Implement GPIO-based method to start the robot
        self._imu.start()  # Start IMU reading thread
        self._left_motor.start()  # Start left motor
        self._right_motor.start()  # Start right motor

    def stop_robot(self):
        # Implement GPIO-based method to stop the robot
        self._imu.stop()  # Stop IMU reading thread
        self._left_motor.stop()  # Stop left motor
        self._right_motor.stop()  # Stop right motor

    def set_speed(self, speed: float):
        # Implement GPIO-based method to set the speed of the robot
        self._left_motor.set_speed(speed)  # Set speed for left motor
        self._right_motor.set_speed(speed)  # Set speed for right motor

    def get_velocity(self):
        # Get velocity of robot in meters/second based on motor speeds and robot kinematics
        motor_speed_rad_per_sec = (self._left_motor.speed + self._right_motor.speed) / 2  # Average speed of both motors
        wheel_radius = .0053  # Wheel radius in meters
        velocity = motor_speed_rad_per_sec * wheel_radius

        return velocity
    
    def get_pitch_velocity(self) -> float:
        # Implement GPIO-based method to read pitch velocity
        # Placeholder implementation
        pitch_velocity = self._imu.pitch_velocity  # Get pitch velocity from IMU
        return pitch_velocity