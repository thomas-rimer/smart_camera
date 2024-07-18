import cv2 as cv
import numpy as np
import pickle
import math

# Parameters
SCAN_ANGLE_FIDELITY = 0.01 # radians
UPDATE_FREQUENCY = 5000 # milliseconds

# Outward communication
def send_indicator_values(indicator_values):
    # This is where the code to send the indicator values to the server will go
    pass

# Read the calibration file
calibration_file_name = 'camera_frame.pkl'
with open(calibration_file_name, 'rb') as inp:
    dial_objects = pickle.load(inp)

# Open the camera
cap = cv.VideoCapture(0) # 0 is iPhone, 1 is built-in webcam

cv.waitKey(5000) # wait for the camera to warm up

# add error handling
if not cap.isOpened():
    print('Error! CHANGE THE CAMERA INDEX TO 0')
    exit()
else:
    print("Camera opened successfully...")

while True:
    # Ingest camera feed
    ret, frame = cap.read() # read a frame from the camera
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert the frame to greyscale
    frame = cv.bitwise_not(frame) # invert the greyscale frame

    indicator_values = []

    for dial in dial_objects:
        # Crop the dial from the frame
        dial_frame = frame[dial.center_coords[1]-dial.get_radius():dial.center_coords[1] + dial.get_radius(), dial.center_coords[0] - dial.get_radius():dial.center_coords[0] + dial.get_radius()]
        
        # Mask the dial
        circlular_mask = np.zeros(dial_frame.shape[:2], dtype='uint8')
        cv.circle(circlular_mask, (dial.get_radius(), dial.get_radius()), dial.get_radius(), 255, -1)
        dial_frame = cv.bitwise_and(dial_frame, dial_frame, mask=circlular_mask)
        
        # Threshold the masked dial
        dial_frame = cv.threshold(dial_frame, dial.threshold_value, 255, cv.THRESH_BINARY)[1]

        # Erode the thresholded dial
        kernel = np.ones((dial.kernel_size, dial.kernel_size), np.uint8)
        dial_frame = cv.erode(dial_frame, kernel, iterations=1)

        # Dilate the eroded dial
        dial_frame = cv.dilate(dial_frame, kernel, iterations=1)

        # Find and select the largest contour in the dial
        contours, _ = cv.findContours(dial_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv.contourArea)
        circlular_mask = np.zeros_like(dial_frame)
        cv.drawContours(circlular_mask, [max_contour], -1, 255, -1)
        dial_frame = cv.bitwise_and(dial_frame, circlular_mask)

        # Get the indicator value
        dial.get_indicator_value(dial_frame)

        # Save to the list
        indicator_values.append([dial.ident, dial.indicator_value])

    for index, indicator_value in enumerate(indicator_values):
        print(f"{dial_objects[index].name}: {indicator_value[1]} {dial_objects[index].units}")
    print("")
    
    send_indicator_values(indicator_values)

    # Check if the user pressed the 'q' key
    if cv.waitKey(UPDATE_FREQUENCY) & 0xFF == ord('q'):
        break