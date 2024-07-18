import cv2 as cv
import numpy as np
import pickle
import math


# Read the image
frame_path = 'camera_frame.png'
original_image = cv.imread(frame_path)

# Read the calibration file
calibration_file_name = frame_path.split('.')[-2] + '.pkl'
with open(calibration_file_name, 'rb') as inp:
    dial_objects = pickle.load(inp)

# Split the image into each dial
dial_images = []
for dial in dial_objects:
    cropped_dial = original_image[dial.center_coords[1]-dial.get_radius():dial.center_coords[1] + dial.get_radius(), dial.center_coords[0] - dial.get_radius():dial.center_coords[0] + dial.get_radius()]
    dial_images.append(cropped_dial)

original_images = dial_images.copy()

# Convert each dial to greyscale
for index, dial_image in enumerate(dial_images):
    dial_images[index] = cv.cvtColor(dial_image, cv.COLOR_BGR2GRAY)

# Invert each greyscale image
for index, dial_image in enumerate(dial_images):
    dial_images[index] = cv.bitwise_not(dial_image)

# Mask each dial
for index, dial_image in enumerate(dial_images):
    mask = np.zeros(dial_image.shape[:2], dtype='uint8')
    cv.circle(mask, (dial_objects[index].get_radius(), dial_objects[index].get_radius()), dial_objects[index].get_radius(), 255, -1)
    dial_images[index] = cv.bitwise_and(dial_image, dial_image, mask=mask)

# Threshold each inverted image
for index, dial_image in enumerate(dial_images):
    dial_images[index] = cv.threshold(dial_image, dial_objects[index].threshold_value, 255, cv.THRESH_BINARY)[1] # value of 150 came from experimenting with main.py

# Erode and dilate each thresholded image
for index, dial_image in enumerate(dial_images):
    kernel = np.ones((dial_objects[index].kernel_size, dial_objects[index].kernel_size), np.uint8)
    dial_images[index] = cv.erode(dial_image, kernel, iterations=1)

# Dilate each eroded image
for index, dial_image in enumerate(dial_images):
    kernel = np.ones((dial_objects[index].kernel_size, dial_objects[index].kernel_size), np.uint8)
    dial_images[index] = cv.dilate(dial_image, kernel, iterations=1)

# Find the largest contour in each dial image and leave only it
for index, dial_image in enumerate(dial_images):
    contours, _ = cv.findContours(dial_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    mask = np.zeros_like(dial_image)
    cv.drawContours(mask, [max_contour], -1, 255, -1)
    dial_images[index] = cv.bitwise_and(dial_image, mask)

# Find the angle of the indicator in each dial image
dial_angles = []
for index, dial_image in enumerate(dial_images):
    # Set up necessary varibales
    max_indicator_value = 0
    max_indicator_angle = 0 # radians
    angle_fidelity = 0.01 # radians

    center_point = (int(dial_objects[index].get_radius()), int(dial_objects[index].get_radius()))

    # Sweep a line across all lines from the center to the circumference
    angle = 0
    while angle < 2 * math.pi:
        # Mask the image with the line
        circumnferemce_point = (int(center_point[0] + math.cos(angle)*dial_objects[index].get_radius()), int(center_point[1] - math.sin(angle)*dial_objects[index].get_radius()))
        mask = np.zeros(dial_image.shape[:2], dtype='uint8')
        cv.line(mask, center_point, circumnferemce_point, color=(255), thickness=3)
        line_pixels = cv.bitwise_and(dial_image, dial_image, mask=mask)
        
        # Calculate the mean intensity and set maximum if it's the largest observed so far
        mean_val = cv.mean(line_pixels, mask=mask)[0]
        if mean_val > max_indicator_value:
            max_indicator_value = mean_val
            max_indicator_angle = angle

        angle += angle_fidelity

    dial_angles.append(max_indicator_angle)


# Calculate actual readout
for index, dial_image in enumerate(dial_images):
    print("")
    print("")
    print('############# Dial Number: ' + str(dial_objects[index].name) + ' #############') 
    print("")
    print("")   

    # Print the original min, max, intermediate tick angles and indicator angle
    print('Original min angle: ' + str(dial_objects[index].get_min_angle()))
    print('Original max angle: ' + str(dial_objects[index].get_max_angle()))
    print('Original intermediate tick angles: ' + str(dial_objects[index].get_intermediate_tick_angles()))
    print('Original indicator angle: ' + str(dial_angles[index]))
    print("")

    # Rotate all angles such that that min is at zero
    modified_min_angle = 0
    modified_max_angle = (dial_objects[index].get_max_angle() - dial_objects[index].get_min_angle())%(2*math.pi)
    modified_intermediate_tick_angles = [(angle - dial_objects[index].get_min_angle())%(2*math.pi) for angle in dial_objects[index].get_intermediate_tick_angles()]
    modified_indicator_angle = (max_indicator_angle - dial_objects[index].get_min_angle())%(2*math.pi)

    # Print the four variables above
    print('Rotated min angle: ' + str(modified_min_angle))
    print('Rotated max angle: ' + str(modified_max_angle))
    print('Rotated intermediate tick angles: ' + str(modified_intermediate_tick_angles))
    print('Rotated indicator angle: ' + str(modified_indicator_angle))
    print("")

    # Flip such that dial always reads CCW
    if modified_intermediate_tick_angles[0] > modified_max_angle:
        modified_max_angle = 2*math.pi - modified_max_angle
        modified_intermediate_tick_angles = [2*math.pi - angle for angle in modified_intermediate_tick_angles]
        modified_indicator_angle = 2*math.pi - modified_indicator_angle

    # Print the four variables above
    print('Flipped min angle: ' + str(modified_min_angle))
    print('Flipped max angle: ' + str(modified_max_angle))
    print('Flipped intermediate tick angles: ' + str(modified_intermediate_tick_angles))
    print('Flipped indicator angle: ' + str(modified_indicator_angle))
    print("")
    
    # Measure without correction
    dial_reported_range = dial_objects[index].max_value - dial_objects[index].min_value
    dial_reading = (modified_indicator_angle/modified_max_angle) * dial_reported_range + dial_objects[index].min_value

    print('Max_value: ' + str(dial_objects[index].max_value))
    print('Min_value: ' + str(dial_objects[index].min_value))
    print('Dial_reported_range: ' + str(dial_reported_range))
    print('Fraction of max angle: ' + str(modified_indicator_angle/modified_max_angle))
    print("")

    print('--- Indicated value without correction ---')
    print('Dial reading: ' + str(dial_reading) + ' ' + str(dial_objects[index].units))
    print("")

    # Measure with correction
    all_ticks = list(zip([modified_min_angle] + modified_intermediate_tick_angles + [modified_max_angle], [dial_objects[index].min_value] + dial_objects[index].intermediate_tick_values + [dial_objects[index].max_value]))
    sorted_all_ticks = sorted(all_ticks, key=lambda x: x[0])
    sorted_all_tick_angles, sorted_all_tick_values = zip(*sorted_all_ticks)
    sorted_all_tick_angles = list(sorted_all_tick_angles)
    sorted_all_tick_values = list(sorted_all_tick_values)

    print('All ticks: ' + str(all_ticks))
    print("")

    # Identify which range the indicator is in
    index = 0
    while index < len(sorted_all_tick_angles) - 1:
        print('Current lower bound tick angle: ' + str(sorted_all_tick_angles[index]))
        print('Current lower bound tick value: ' + str(sorted_all_tick_values[index]))
        print('Current upper bound tick angle: ' + str(sorted_all_tick_angles[index + 1]))
        print('Current upper bound tick value: ' + str(sorted_all_tick_values[index + 1]))
        print("")
        if modified_indicator_angle < sorted_all_tick_angles[index + 1]:
            # indicator is in between current tick and next tick
            frac_into_section = (modified_indicator_angle - sorted_all_tick_angles[index])/(sorted_all_tick_angles[index+1] - sorted_all_tick_angles[index])
            corrected_dial_reading = frac_into_section * (sorted_all_tick_values[index+1] - sorted_all_tick_values[index]) + sorted_all_tick_values[index]

            print('--- Indicated value WITH correction ---')
            print('###################################################')
            print('###################################################')
            print('############    ' + str(corrected_dial_reading) + '    ############')
            print('###################################################')
            print('###################################################')
            print("")

        index += 1
    

# Display an overlay on the original dials
for index, dial_image in enumerate(original_images):
    center_point = (int(dial.get_radius()), int(dial.get_radius()))
    circumnferemce_point = (int(center_point[0] + math.cos(dial_angles[index])*dial.get_radius()), int(center_point[1] - math.sin(dial_angles[index])*dial.get_radius()))
    cv.line(dial_image, center_point, circumnferemce_point, color=(0, 0, 255), thickness=5)
    cv.imshow('dial ' + str(index), dial_image)

cv.waitKey(0)
cv.destroyAllWindows()