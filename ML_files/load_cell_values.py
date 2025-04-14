import serial
import time

# Update the port to match your Arduino's port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
arduino_port = "COM3"  # Replace with your port
baud = 9600
timeout = 2

try:
    ser = serial.Serial(arduino_port, baud, timeout=timeout)
    time.sleep(2)  # wait for Arduino to reset

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(f"Weight: {line} g")
except serial.SerialException as e:
    print(f"Serial error: {e}")
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    if 'ser' in locals():
        ser.close()
