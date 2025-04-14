import serial
import time

# Set up the serial port (Adjust the port to match your system)
arduino_port = "COM3"  # Use the correct port for your system
baud_rate = 9600       # Baud rate should match Arduino's setting
timeout = 1            # Timeout for serial communication

# Open serial connection
ser = serial.Serial(arduino_port, baud_rate, timeout=timeout)

time.sleep(2)  # Give Arduino time to reset

previous_weight = None  # Variable to store the previous weight value

try:
    while True:
        if ser.in_waiting > 0:  # Check if data is available to read
            line = ser.readline().decode('utf-8').strip()  # Read and decode the data
            if line:
                current_weight = int(line)  # Convert to integer
                if current_weight != previous_weight:  # Only print if weight changed
                    print(f"Current Weight: {current_weight} g")
                    previous_weight = current_weight  # Update previous weight
except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    ser.close()  # Close the serial port when done
