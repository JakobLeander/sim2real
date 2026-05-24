"""
Stepper Motor interface for controlling a stepper motor.

# Tested with motor SM-42BYG011-25 from Mercury using a DRV8825 stepper driver.
# Using pippio hardware PWM for better performance and smoother motion.
# The pigpio daemon must be running
# sudo pigpiod

"""

import serial
import time
import struct
import threading
import RPi.GPIO as GPIO
import time
import sys
import math
import pigpio

# Degrees per step for motor is 1.8 degree  with 1/4 microstepping, we get 0.45 degree per step
RAD_PER_STEP = 1.8 * math.pi / 180/4  # 4 microsteps per step (1/4 microstepping)

# Max speed in rad/second
MAX_SPEED = 20.0


class StepperMotor:
    """Stepper Motor class to interface with the stepper motor."""

    def __init__(self, step_pin: int, dir_pin:int, enable_pin: int, reverse_direction: bool = False):
        """Initialize the Stepper Motor with the given GPIO pins."""
        self._step_pin = step_pin
        self._dir_pin = dir_pin
        self._enable_pin = enable_pin
        self._reverse_direction = reverse_direction
        self.speed=0.0

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running")

        #setup pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._dir_pin, GPIO.OUT)
        GPIO.setup(self._enable_pin, GPIO.OUT)
        self.pi.set_mode(self._step_pin, pigpio.OUTPUT)
        
        # Initialize motor as off
        GPIO.output(self._enable_pin, GPIO.HIGH) #turned off

    # ==================== PUBLIC API ====================

    def start(self):
        """Start the motor"""
        GPIO.output(self._enable_pin, GPIO.LOW)  # Enable the motor

    def stop(self):
        """Stop the motor."""
        GPIO.output(self._enable_pin, GPIO.HIGH)  # Disable the motor

    def set_speed(self, speed_rad_per_sec):
        # Clamp speed
        speed = max(-MAX_SPEED, min(speed_rad_per_sec, MAX_SPEED))
        self.speed = speed

        # Set direction
        direction = 1 if speed >= 0 else -1
        if self._reverse_direction:
            direction *= -1

        self.pi.write(self._dir_pin, 1 if direction > 0 else 0)

        # Convert rad/s → steps/s
        abs_speed = abs(speed)
        if abs_speed == 0:
            self.pi.hardware_PWM(self._step_pin, 0, 0)
            return

        steps_per_sec = abs_speed / RAD_PER_STEP

        # pigpio hardware PWM:
        # frequency = steps per second
        # dutycycle = 500000 (50%)
        # TODO: Likely add acceleration ramping for smoother motion using a separate thread to update the speed gradually instead of instant changes
        # might not be needed for robot with microstepping and relatively low motor speeds
        self.pi.hardware_PWM(self._step_pin, int(steps_per_sec), 500000)

def main():
    """Main function for testing the stepper motor."""

    ENABLE_PIN = 21
    STEP_PIN = 12
    DIR_PIN = 20

    ENABLE_PIN = 26
    STEP_PIN = 19
    DIR_PIN = 13
    SPEED = math.pi*2  # 1 rotation per second
    

    motor = StepperMotor(STEP_PIN, DIR_PIN, ENABLE_PIN)

    motor.start()
    motor.set_speed(SPEED)  # Set speed to 5 rad/s
    time.sleep(5)
    motor.set_speed(-SPEED)  # Set speed to 5 rad/s
    time.sleep(5)
    motor.stop()

if __name__ == "__main__":
    main()