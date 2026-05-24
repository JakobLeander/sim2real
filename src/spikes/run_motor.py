#!/usr/bin/env python3
"""
Stepper Motor Control Script

This script provides basic control for a stepper motor using Raspberry Pi GPIO.
The step pin is connected to GPIO 20 as requested.
# Tested with motor SM-42BYG011-25
"""

import RPi.GPIO as GPIO
import time
import sys
import pigpio

# Pin definitions
STEP_PIN = 12      # GPIO 12 - Step signal (user requested)
ENABLE_PIN = 21

# Motor parameters
DEGREES_PER_STEP = 1.8/4       # Degrees per step for a typical 200 steps/rev motor with microstepping 1/4

class StepperMotor:
    """
    Class to control a stepper motor via GPIO pins.
    """

    def __init__(self, step_pin=STEP_PIN, enable_pin=ENABLE_PIN):
        """
        Initialize the stepper motor controller.

        Args:
            step_pin (int): GPIO pin for step signal
            dir_pin (int): GPIO pin for direction signal
            enable_pin (int): GPIO pin for enable signal (active low)
        """
        self.step_pin = step_pin
        self._enable_pin = enable_pin
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.step_pin, GPIO.OUT)
        GPIO.setup(self._enable_pin, GPIO.OUT)
        
        # Initialize pins
        GPIO.output(self.step_pin, GPIO.LOW)
        GPIO.output(self._enable_pin, GPIO.LOW)  # Enable pin is active low
        
        print(f"Stepper motor initialized on pins: STEP={step_pin}, ENABLE={enable_pin}")
    
    def step(self, steps, delay):
        """
        Move the motor a specified number of steps.

        Args:
            steps (int): Number of steps to move
            delay (float): Delay between steps in seconds (half the total step period)
        """

        for i in range(steps):
            # Send step pulse - high for delay, then low for delay
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(delay)


        print(f"Moved {steps} steps )")
   
    def cleanup(self):
        GPIO.output(self._enable_pin, GPIO.HIGH)  # Enable pin is active low

        """Clean up GPIO pins."""
        GPIO.cleanup()
        print("GPIO cleanup completed")


def main():
    """Main function for testing the stepper motor."""

    motor = StepperMotor()

    desired_speed_rpm = 1  # Desired speed in RPM
    steps_per_revolution = 360 / DEGREES_PER_STEP  # Calculate steps per revolution
    delay_between_steps = 1.0 / (desired_speed_rpm * steps_per_revolution)
    
    # Since we have sleep(delay) for HIGH and sleep(delay) for LOW,
    # the total time per step is 2 * delay, so we need delay = delay_between_steps / 2
    step_delay = delay_between_steps / 2

    print(f"Calculated step delay for {desired_speed_rpm} RPM: {step_delay:.4f} seconds per half step")
    
    try:
        print("Running stepper motor test sequence...")

        # Test basic stepping
        print("\nTesting clockwise rotation...")
        motor.step(500, step_delay)

        time.sleep(1)

    finally:
        motor.cleanup()


if __name__ == "__main__":
    main()