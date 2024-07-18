import cv2 as cv
import numpy as np

# Write a simple script that captures a single frame from the webcam and saves it to disk

# Open the camera
cap = cv.VideoCapture(0) # 0 is iPhone, 1 is built-in webcam

# add error handling
if not cap.isOpened():
    print('Error! CHANGE THE CAMERA INDEX TO 0')
    exit()

cv.waitKey(5000) # wait for the camera to warm up
# Ingest camera feed
ret, frame = cap.read() # read a frame from the camera

# Save the frame to disk
cv.imwrite('camera_frame.png', frame)