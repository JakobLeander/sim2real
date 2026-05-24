import serial
import time
import struct
import math

# -----------------------------
# UART SETUP
# -----------------------------
print("[DBG] Opening UART...")
ser = serial.Serial(
    port="/dev/serial0",
    baudrate=115200,
    timeout=0.1
)

time.sleep(0.7)
ser.reset_input_buffer()
print("[DBG] UART ready")

# -----------------------------
# UART PROTOCOL HELPERS
# -----------------------------
def bno_write(reg, value):
    print(f"[DBG] WRITE reg=0x{reg:02X}, value=0x{value:02X}")
    packet = bytes([0xAA, 0x00, reg, 0x01, value])
    ser.write(packet)
    ser.flush()
    time.sleep(0.02)

    # Some BNO055 UART implementations return an acknowledgement packet.
    # Read it if available, but do not fail if no response is received.
    ack = ser.read(2)
    if len(ack) == 2 and ack[0] == 0xBB:
        print(f"[DBG]   ACK: {ack}")


def bno_read(reg, length):
    ser.write(bytes([0xAA, 0x01, reg, length]))
    ser.flush()
    time.sleep(0.02)

    header = ser.read(2)

    if len(header) != 2 or header[0] != 0xBB:
        print("[ERR]   Invalid header")
        return None

    reported_length = header[1]
    if reported_length != length:
        print(f"[WARN]   Response length {reported_length} != requested {length}")

    data = ser.read(reported_length)

    if len(data) != reported_length:
        print("[ERR]   Incomplete data")
        return None

    return data

# -----------------------------
# EULER READ
# -----------------------------
def read_euler():
    """Read Euler heading, roll, and pitch from the BNO055.

    The BNO055 returns Euler angles at register 0x1A in little-endian
    signed 16-bit values with a scale of 1/16 degree.
    """
    data = bno_read(0x1A, 6)
    if data is None:
        print("[ERR] Euler read failed")
        return None

    heading, roll, pitch = struct.unpack("<hhh", data)
    scale = 1.0 / 16.0
    return heading * scale, roll * scale, pitch * scale

def read_calib():
    data = bno_read(0x35, 1)
    if data is None:
        return None

    calib = data[0]
    acc =  calib       & 0x03   # bits 0-1
    mag = (calib >> 2) & 0x03   # bits 2-3
    gyr = (calib >> 4) & 0x03   # bits 4-5
    sys = (calib >> 6) & 0x03   # bits 6-7

    return acc, mag, gyr, sys


# -----------------------------
# INITIALIZATION WITH DEBUG
# -----------------------------
print("[DBG] Switching to CONFIGMODE")
bno_write(0x3D, 0x00)
time.sleep(0.02)

print("[DBG] Setting power mode NORMAL")
bno_write(0x3E, 0x00)

print("[DBG] Clearing SYS_TRIGGER")
bno_write(0x3F, 0x00)

print("[DBG] Selecting PAGE 0")
bno_write(0x07, 0x00)

# Apply remap
bno_write(0x41, 0x06)   # X=Z, Y=Y, Z=X
bno_write(0x42, 0x02)   # Flip Y

print("[DBG] Switching to NDOF mode")
bno_write(0x3D, 0x0C)
time.sleep(1.0)

print("Checking NDOF mode...")
print(bno_read(0x3D, 1))


print("[DBG] Initialization complete")

# -----------------------------
# MAIN LOOP
# -----------------------------
print("[DBG] Starting Euler read loop...")
print(bno_read(0x3D, 1))

# check if it is calibrated
calib = read_calib()
if calib is not None:
    acc_calib, mag_calib, gyr_calib, sys_calib = calib
    print(f"[DBG] Calibration status: Acc={acc_calib}, Mag={mag_calib}, Gyr={gyr_calib}, Sys={sys_calib}")
else:
    print("[ERR] Failed to read calibration status")


while True:
    euler = read_euler()
    if euler is not None:
        heading, roll, pitch = euler
        print(f"[DBG] Euler angles: Heading={heading:+07.2f}°, Roll={roll:+07.2f}°, Pitch={pitch:+07.2f}°")
    else:
        print("[ERR] Failed to read Euler angles")
    time.sleep(0.2)
