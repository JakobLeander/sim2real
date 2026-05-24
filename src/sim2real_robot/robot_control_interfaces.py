"""
Robot control interfaces for sim2real robot systems.

This module defines abstract interfaces for robot control systems,
providing a standardized way to interact with different robot implementations
in simulation and real-world environments.
"""

from abc import ABC, abstractmethod

class RobotControl(ABC):
    """
    Abstract base class for robot control interfaces.

    This interface defines the contract that all robot control implementations
    must follow. It provides methods for accessing robot state information
    such as orientation and position data.
    """

    @abstractmethod
    def get_pitch_angle(self) -> float:
        """
        Get the current pitch angle of the robot.

        Returns:
            float: The pitch angle in radians, where positive values indicate
                   forward tilt and negative values indicate backward tilt.
        """
        pass

    @abstractmethod
    def get_pitch_velocity(self) -> float:
        """
        Get the current pitch velocity of the robot.

        Returns:
            float: The pitch velocity in radians/second, where positive values indicate
                   increasing forward tilt and negative values indicate increasing backward tilt.
        """
        pass

    @abstractmethod
    def start_robot(self):
        """
        Start the robot.
        This method initializes the robot and prepares it for operation.
        """
        pass

    @abstractmethod
    def stop_robot(self):
        """
        Stop the robot.
        This method shuts down the robot and releases any resources it is using.
        """
        pass

    @abstractmethod
    def set_speed(self, speed: float):
        """
        Set the speed of the robot in rad/second.

        Args:
            speed (float): The speed at which to operate the robot.
        """
        pass

    @abstractmethod
    def get_velocity(self)-> float:
        """
        Get the current velocity of the robot in meters/second
        """
        pass
