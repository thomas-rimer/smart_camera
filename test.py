import cv2 as cv

example_threshold_path = 'example_threshold_dials.png'

def set_threshold_intro_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.destroyWindow('Example of good and bad threshold values')
        print("HELLO")
        exit()

def set_threshold(event, x, y, flags, param):
    pass

image_path = 'example_threshold_dials.png'
cv.namedWindow('Example of good and bad threshold values') # create a new window
cv.setMouseCallback('Example of good and bad threshold values', set_threshold_intro_callback)

while True:
    cv.imshow('Example of good and bad threshold values', cv.imread(example_threshold_path))
    key = cv.waitKey(1)