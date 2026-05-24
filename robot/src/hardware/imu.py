"""
IMU (Inertial Measurement Unit) interface for BNO055 sensor.

This module provides an interface to the BNO055 9-DOF IMU sensor via UART.
Tested with Fermion BNO055 IMU from DFRobot.
"""

import serial
import time
import struct
import threading


class IMU:
    """IMU class to interface with the BNO055 sensor via UART."""

    def __init__(self, port="/dev/serial0", baudrate=115200):
        """Initialize the IMU with the given UART port."""
        self.available = False
        self._thread = None
        self._running = False
        
        # Cached sensor data
        self.pitch_velocity_deg_s = 0.0  # Cache for pitch angular velocity
        self.pitch_angle_deg = 0.0  # Cache for pitch angular velocity
        
        # Reading rate tracking
        self.readings_per_second = 0
        
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=0.1
            )
            time.sleep(0.7)
            
            self.ser.reset_input_buffer()
            self._initialize()
            self.available = True
        except Exception as e:
            print(f"Warning: Could not initialize BNO055 IMU: {e}")
            self.available = False

    # ==================== PUBLIC API ====================

    def start(self):
        """Start the background thread for continuous sensor reading."""
        if self._running:
            print("[WARN] IMU thread already running")
            return
        
        if not self.available:
            print("[ERR] Cannot start IMU thread: IMU not available")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[DBG] IMU reading thread started")

    def stop(self):
        """Stop the background thread."""
        if not self._running:
            return
        
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        print("[DBG] IMU reading thread stopped")

    def get_calibration_status(self):
        """Get the calibration status of the IMU."""
        if not self.available:
            return None

        try:
            data = self._read(0x35, 1)
            if data is not None:
                calib = data[0]
                acc = calib & 0x03  # bits 0-1
                mag = (calib >> 2) & 0x03  # bits 2-3
                gyr = (calib >> 4) & 0x03  # bits 4-5
                sys = (calib >> 6) & 0x03  # bits 6-7
                return acc, mag, gyr, sys
        except Exception as e:
            print(f"Warning: Could not read calibration status from IMU: {e}")
        return None

    # ==================== PRIVATE IMPLEMENTATION ====================

    def _write(self, reg, value):
        """Write a register to the BNO055."""
        if not self.available:
            return
        packet = bytes([0xAA, 0x00, reg, 0x01, value])
        self.ser.write(packet)
        self.ser.flush()

    def _read(self, reg, length):
        """Read registers from the BNO055."""
        if not self.available:
            return None
        self.ser.write(bytes([0xAA, 0x01, reg, length]))

        time.sleep(0.005)

        header = self.ser.read(2)
        if len(header) != 2 or header[0] != 0xBB:
            # means we failed to get value, this happens frequently in bno055
            # if this happens we will return old values, which is better than crashing the program
            return None

        reported_length = header[1]
        if reported_length != length:
            print(f"[WARN]   Response length {reported_length} != requested {length}")

        data = self.ser.read(reported_length)

        if len(data) != reported_length:
            print("[ERR]   Incomplete data")
            return None

        return data

    def _initialize(self):
        """Initialize the BNO055 sensor in NDOF mode."""
        # Soft reset
        self._write(0x3F, 0x20)
        time.sleep(0.65)  # allow reboot
        
        # Switching to CONFIGMODE
        self._write(0x3D, 0x00)
        time.sleep(0.05)

        #Selecting PAGE 0
        self._write(0x07, 0x00)
        time.sleep(0.05)

        # Setting power mode NORMAL
        self._write(0x3E, 0x00)
        time.sleep(0.05)

        # Clearing SYS_TRIGGER
        self._write(0x3F, 0x00)
        time.sleep(0.05)


        # Apply remap
        self._write(0x41, 0x06)   # X=Z, Y=Y, Z=X
        time.sleep(0.05)
        self._write(0x42, 0x02)   # Flip Y
        time.sleep(0.05)

        # Switching to NDOF mode
        self._write(0x3D, 0x0C)
        time.sleep(1.0)

    def _read_loop(self):
        """Background thread that continuously reads sensor data."""
        read_count = 0
        current_time = time.time()
        last_rate_time = current_time
        elapsed = 0.0
        time_between_reads = 0.01  # Target 100 Hz update rate

        while self._running:
            try:
                current_read_begin = time.time()    
                # read data in one operation to make it go faster
                data = self._read(0x16, 8)
                if data is not None:
                    gyr_y_raw = struct.unpack_from("<h", data, 0)[0]
                    pitch_raw = struct.unpack_from("<h", data, 6)[0]
                    self.pitch_angle_deg = pitch_raw / 16.0
                    self.pitch_velocity_deg_s = gyr_y_raw / 16.0
                
                # Track reading rate
                read_count += 1
                current_time = time.time()
                elapsed = current_time - last_rate_time
                if elapsed >= 1.0:
                    self.readings_per_second = read_count / elapsed
                    read_count = 0
                    last_rate_time = current_time
                
                current_read_end = time.time()    
                current_read_duration = current_read_end - current_read_begin
                sleep_time = time_between_reads - current_read_duration
                if sleep_time < 0:
                    sleep_time = 0

                time.sleep(sleep_time)  # 100 Hz update rate
            except Exception as e:
                print(f"[ERR] Error in IMU read loop: {e}")
                time.sleep(0.001)


if __name__ == "__main__":
    # Initialize the IMU sensor
    imu = IMU()

    if not imu.available:
        print("[ERR] IMU not available. Exiting.")
        exit(1)

    # Check calibration status
    calib = imu.get_calibration_status()
    if calib is not None:
        acc_calib, mag_calib, gyr_calib, sys_calib = calib
        print(f"[DBG] Calibration status: Acc={acc_calib}, Mag={mag_calib}, Gyr={gyr_calib}, Sys={sys_calib}")
    else:
        print("[ERR] Failed to read calibration status")

    # Start the background reading thread
    imu.start()

    # Main loop: read and print Euler angles and angular velocity
    print("[DBG] Starting main read loop...")
    try:
        while True:
            pitch = imu.pitch_angle_deg
            pitch_angular_velocity = imu.pitch_velocity_deg_s
            readings_per_second = imu.readings_per_second
            print(f"Pitch: Angle={pitch:.2f}°, AngularVel={pitch_angular_velocity:.2f}°/s, Rate={readings_per_second:.1f} Hz")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[DBG] Exiting...")
        imu.stop()

