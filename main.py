# import required libraries
import cv2 as cv
import numpy as np
import pickle

def nothing(x):
    pass


# Read the image
image_name = 'dials'
original_image = cv.imread(image_name + '.png')
cv.imshow('Original Image', original_image)

# Read the calibration file
file_name = image_name + '.pkl'
with open(file_name, 'rb') as inp:
    dials = pickle.load(inp)

# Mask the image with a circle around the dial
mask = np.zeros(original_image.shape[:2], dtype='uint8')
for dial in dials:
    cv.circle(mask, dial.center_coords, dial.calculate_average_radius(), 255, -1)

masked_image = cv.bitwise_and(original_image, original_image, mask=mask)
cv.imshow('Masked Image', masked_image)

# Crop image to the masked region
x, y, w, h = cv.boundingRect(mask)
masked_image = masked_image[y:y+h, x:x+w]
cv.imshow('Cropped Image', masked_image)

# Convert to grayscale
gray_image = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray_image)

# Invert the greyscale image
inverted_image = cv.bitwise_not(gray_image)
cv.imshow('Inverted Gray Image', inverted_image)

cv.namedWindow('Thresholded Image')
cv.createTrackbar('threshold', 'Thresholded Image', 0, 255, nothing)

while(True):
    # Display thresholded image
    thresholded_image = cv.threshold(inverted_image, cv.getTrackbarPos('threshold', 'Thresholded Image'), 255, cv.THRESH_BINARY)[1]
    cv.imshow('Thresholded Image', thresholded_image)

    # Display eroded image
    kernel = np.ones((5,5), np.uint8)
    eroded_image = cv.erode(thresholded_image, kernel, iterations=1)
    cv.imshow('Eroded Image', eroded_image)

    # Display dilated image
    dilated_image = cv.dilate(eroded_image, kernel, iterations=1)
    cv.imshow('Dilated Image', dilated_image)

    # For each dial, detect the largest contour, and mask it
    for dial in dials:
        contours, _ = cv.findContours(dilated_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)
        mask = np.zeros(dilated_image.shape, dtype='uint8')
        cv.drawContours(mask, [largest_contour], -1, 255, -1)
        masked_image = cv.bitwise_and(dilated_image, dilated_image, mask=mask)
        cv.imshow('Masked Image', masked_image)


    if cv.waitKey(1000) & 0xFF == ord('q'):
        break