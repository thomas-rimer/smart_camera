import cv2 as cv
import numpy as np
from dial import Dial
import pickle

# Read the image
frame_path = 'camera_frame.png'
original_frame = cv.imread(frame_path)
height, width, _ = original_frame.shape

# Set up variables for tracking dials
dial_ident = 0 # number used to identify each dial, from 0 to [number of dials in frame]. assigned sequentially to each dial as it is created
dial_objects = [] # list of Dial objects saved after calibration

# Set up variables for instructions
FONT = cv.FONT_HERSHEY_SIMPLEX
MAX_INSTRUCTION_LENGTH = 48
font_size = 20
font_weight = 25

# Set maximum font size and weight based on camera resolution
while cv.getTextSize(" " * MAX_INSTRUCTION_LENGTH, FONT, font_size, font_weight)[0][0] > 0.75 * width:
    font_size = font_size * 0.9
    font_weight = int(font_weight * 0.95)

# Create a display image with a blank header area for instructions
HEADER_HEIGHT = cv.getTextSize(" " * MAX_INSTRUCTION_LENGTH, FONT, font_size, font_weight)[0][1] * 4 # height in pixels of the header space
HEADER_CENTER_COORDS = width//2, HEADER_HEIGHT//2 
display_frame = np.ones((height + HEADER_HEIGHT, width, 3), dtype=np.uint8) * 255
display_frame[HEADER_HEIGHT:, :] = original_frame

# Variables for the exit button
BUTTON_BUFFER = width//50
BUTTON_FONT_SIZE = font_size*0.5
BUTTON_FONT_WEIGHT = int(font_weight*0.5)
button_text = "Save and Exit"
button_textsize = cv.getTextSize(button_text, FONT, BUTTON_FONT_SIZE, BUTTON_FONT_WEIGHT)[0]

# Paramters for misc drawing features
DOT_RADIUS = width//250
LINE_STROKE_WEIGHT = DOT_RADIUS//2
INTERMEDIATE_TICK_LEGTH = DOT_RADIUS

# Set up variables for the example thresholded image
EXAMPLE_THRESHOLD_PATH = 'example_threshold_dials.png'

# --- DRAWING FUNCTIONS ---
# These functions place features on the display image

def show_instructions(input_text, window='Calibration Window', frame=display_frame): # writes text instructions for the user on the top of a window
    cv.rectangle(frame, (0, 0), (width, HEADER_HEIGHT), (255, 255, 255), -1) # cover previous instructions with white rectangle
    text_size = cv.getTextSize(input_text, FONT, font_size, font_weight)[0]
    cv.putText(frame, input_text, (HEADER_CENTER_COORDS[0] - text_size[0]//2, HEADER_CENTER_COORDS[1] + text_size[1]//2), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_weight, cv.LINE_AA)

def show_label(input_text, x, y, window='Calibration Window', frame=display_frame): # adds a text label anywhere on a window
    text_size = cv.getTextSize(input_text, FONT, font_size, font_weight)[0]
    cv.rectangle(frame, (x + 20, y), (x + 20 + text_size[0], y - text_size[1]), (255, 255, 255), -1)
    cv.putText(frame, input_text, (x + 20, y), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_weight, cv.LINE_AA)

def show_button(input_text, window='Calibration Window', frame=display_frame): # shows the "exit and save" button
    text_size = cv.getTextSize(input_text, FONT, BUTTON_FONT_SIZE, BUTTON_FONT_WEIGHT)[0]
    cv.rectangle(frame, (width - text_size[0] - 3 * BUTTON_BUFFER, HEADER_HEIGHT//2 - text_size[1]//2 - BUTTON_BUFFER), (width - BUTTON_BUFFER, HEADER_HEIGHT//2 + text_size[1]//2 + BUTTON_BUFFER), (0, 255, 0), -1)
    cv.putText(frame, input_text, (width - text_size[0] - 2 * BUTTON_BUFFER, HEADER_HEIGHT//2 + text_size[1]//2), cv.FONT_HERSHEY_SIMPLEX, BUTTON_FONT_SIZE, (0, 0, 0), BUTTON_FONT_WEIGHT, cv.LINE_AA)

# --- HELPER FUNCTIONS ---

def save_to_file(): # save to file with same name as the input image
    export_file_name = frame_path.split('/')[-1]
    export_file_name = export_file_name.split('.')[0]
    export_file_name = export_file_name + '.pkl'
    with open(export_file_name, 'wb') as outp:
        pickle.dump(dial_objects, outp, pickle.HIGHEST_PROTOCOL)

def nothing(x): 
    pass

# --- CALLBACK FUNCTIONS ---
# These functions are called when the user clicks on the calibration window
# The callback function is changed after each step of the calibration process

def intro(event, x, y, flags, param): # intro screen shown at the start of the calibration process
    if event == cv.EVENT_LBUTTONDOWN:
        
        # proceed to draw center
        cv.setMouseCallback('Calibration Window', select_center)
        show_instructions("Click on the center of a new dial")
        cv.imshow('Calibration Window', display_frame)

def select_center(event, x, y, flags, param): # user clicks on the center of a new dial or saves and exits
    if event == cv.EVENT_LBUTTONDOWN:
        if x > width - button_textsize[0] - 3 * BUTTON_BUFFER and x < width - BUTTON_BUFFER and y > HEADER_HEIGHT//2 - button_textsize[1]//2 - BUTTON_BUFFER and y < HEADER_HEIGHT//2 + button_textsize[1]//2 + BUTTON_BUFFER:
            # Button was clicked. Save everything to file and close out the program.
            show_instructions("Calibration finished!")
            cv.imshow('Calibration Window', display_frame)
            print("Calibration finished! The dial calibration values have been saved to file.")
            print("Below are the calibration values for each dial:")
            print(" ")
            for dial in dial_objects:
                print(f"##### Dial {dial.ident} #####")
                print(f"Units: {dial.units}")
                print(f"Center coords: {dial.center_coords}")
                print(f"Min dial value: {dial.min_value}")
                print(f"Max dial value: {dial.max_value}")
                print(" ")

            save_to_file()
            exit()
        else:
            # User is creating a new dial
            dial_objects.append(Dial(dial_ident, center_coords=(x, y - HEADER_HEIGHT)))  # save a new dial to the list
            cv.circle(display_frame, (x, y), DOT_RADIUS, (0, 0, 255), -1) # draw red circle at center of dial
            show_label(str(dial_ident), x, y) # label center of dial
            
            # Proceed to draw min
            cv.setMouseCallback('Calibration Window', select_min_tick)
            show_instructions("Click on the min value tick mark of the same dial")
            cv.imshow('Calibration Window', display_frame)
    if event == cv.EVENT_MOUSEMOVE:
        temp_display_frame = display_frame.copy()
        cv.circle(temp_display_frame, (x, y), DOT_RADIUS, (0, 0, 255), -1)
        cv.imshow('Calibration Window', temp_display_frame)

        # Show a zoomed in region
        zoomed_region = original_frame[y - 100 - HEADER_HEIGHT:y + 100 - HEADER_HEIGHT, x - 100:x + 100].copy()
        cv.line(zoomed_region, (100, 0), (100, 200), (0, 0, 255), 1)
        cv.line(zoomed_region, (0, 100), (200, 100), (0, 0, 255), 1)
        cv.imshow('Close-View', zoomed_region)

def select_min_tick(event, x, y, flags, param): # user clicks on the min value tick mark of a new dial
    if event == cv.EVENT_LBUTTONDOWN:
        dial_objects[-1].min_coords = (x, y - HEADER_HEIGHT)
        cv.circle(display_frame, (x, y), DOT_RADIUS, (0, 255, 0), -1) # draw green circle
        cv.line(display_frame, (x, y), (dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0], dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]), (0, 255, 0), LINE_STROKE_WEIGHT) # draw green line from center to min point

        # proceed to draw max
        cv.setMouseCallback('Calibration Window', select_max_tick)
        show_instructions("Click on the max value tick mark of the same dial")
        cv.imshow('Calibration Window', display_frame)
    if event == cv.EVENT_MOUSEMOVE:
        temp_display_frame = display_frame.copy()
        cv.circle(temp_display_frame, (x, y), DOT_RADIUS, (0, 255, 0), -1)
        cv.line(temp_display_frame, (x, y), (dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0], dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]), (0, 255, 0), LINE_STROKE_WEIGHT)
        cv.imshow('Calibration Window', temp_display_frame)

        # Show a zoomed in region
        zoomed_region = original_frame[y - 100 - HEADER_HEIGHT:y + 100 - HEADER_HEIGHT, x - 100:x + 100].copy()
        cv.line(zoomed_region, (100, 0), (100, 200), (0, 0, 255), 1)
        cv.line(zoomed_region, (0, 100), (200, 100), (0, 0, 255), 1)
        cv.imshow('Close-View', zoomed_region)

def select_max_tick(event, x, y, flags, param): # user clicks on the max value tick mark of a new dial
    if event == cv.EVENT_LBUTTONDOWN:
        dial_objects[-1].max_coords = (x, y - HEADER_HEIGHT)
        cv.circle(display_frame, (x, y), DOT_RADIUS, (0, 255, 0), -1) # draw green circle
        cv.line(display_frame, (x, y), (dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0], dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]), (0, 255, 0), LINE_STROKE_WEIGHT) # draw green line from center to max point
        cv.circle(display_frame, dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT), dial_objects[-1].get_radius(), (0, 255, 0), LINE_STROKE_WEIGHT)
        
        # proceed to draw upper middle
        cv.setMouseCallback('Calibration Window', select_intermediate_ticks)
        show_instructions("Click at least 3 intermediate ticks, enter their values")
        cv.imshow('Calibration Window', display_frame)
    if event == cv.EVENT_MOUSEMOVE:
        temp_display_frame = display_frame.copy()
        cv.circle(temp_display_frame, (x, y), DOT_RADIUS, (0, 255, 0), -1)
        cv.line(temp_display_frame, (x, y), (dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0], dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]), (0, 255, 0), LINE_STROKE_WEIGHT)
        cv.imshow('Calibration Window', temp_display_frame)

        # Show a zoomed in region
        zoomed_region = original_frame[y - 100 - HEADER_HEIGHT:y + 100 - HEADER_HEIGHT, x - 100:x + 100].copy()
        cv.line(zoomed_region, (100, 0), (100, 200), (0, 0, 255), 1)
        cv.line(zoomed_region, (0, 100), (200, 100), (0, 0, 255), 1)
        cv.imshow('Close-View', zoomed_region)

def select_intermediate_ticks(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if x > width - button_textsize[0] - 3 * BUTTON_BUFFER and x < width - BUTTON_BUFFER and y > HEADER_HEIGHT//2 - button_textsize[1]//2 - BUTTON_BUFFER and y < HEADER_HEIGHT//2 + button_textsize[1]//2 + BUTTON_BUFFER:
            # Button was clicked. User is finished adding intermediate ticks
            
            # Proceed to set threshold intro
            cv.setMouseCallback('Calibration Window', set_threshold_intro)
            show_instructions("Click anywhere to continue to set the threshold...")
            cv.imshow('Calibration Window', display_frame)
        else:
            # User is adding a new intermediate tick

            # Draw a line from the center of the circle to the mouse position BUT it should only have length dial_objects[-1].get_radius()
            x_diff = x - dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0]
            y_diff = y - dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]
            angle = np.arctan2(y_diff, x_diff)

            circ_x = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0] + int(dial_objects[-1].get_radius() * np.cos(angle))
            circ_y = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1] + int(dial_objects[-1].get_radius() * np.sin(angle))

            x_inner = circ_x - int(INTERMEDIATE_TICK_LEGTH * np.cos(angle))
            y_inner = circ_y - int(INTERMEDIATE_TICK_LEGTH * np.sin(angle))

            x_outer = circ_x + int(INTERMEDIATE_TICK_LEGTH * np.cos(angle))
            y_outer = circ_y + int(INTERMEDIATE_TICK_LEGTH * np.sin(angle))

            dial_objects[-1].intermediate_tick_coords.append([circ_x, circ_y - HEADER_HEIGHT])
            cv.line(display_frame, (x_inner, y_inner), (x_outer, y_outer), (255, 125, 0), LINE_STROKE_WEIGHT)
            
            show_instructions("Switch to terminal to enter value...")
            cv.imshow('Calibration Window', display_frame)
            cv.waitKey(50)

            tick_mark_value = input("Enter the value of the tick mark: ")
            print("Return to the opencv window to continue adding intermediate ticks...")
            print("")
            dial_objects[-1].intermediate_tick_values.append(float(tick_mark_value))
            
            if len(dial_objects[-1].intermediate_tick_values) >= 3:
                show_button("Finish adding")
                cv.imshow('Calibration Window', display_frame)
    if event == cv.EVENT_MOUSEMOVE:
        temp_display_frame = display_frame.copy()

        # draw a line from the center of the circle to the mouse position BUT it should only have length dial_objects[-1].get_radius()
        x_diff = x - dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0]
        y_diff = y - dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1]
        angle = np.arctan2(y_diff, x_diff)

        x_inner = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0] + int((dial_objects[-1].get_radius() - INTERMEDIATE_TICK_LEGTH) * np.cos(angle))
        y_inner = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1] + int((dial_objects[-1].get_radius() - INTERMEDIATE_TICK_LEGTH) * np.sin(angle))

        x_outer = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[0] + int((dial_objects[-1].get_radius() + INTERMEDIATE_TICK_LEGTH) * np.cos(angle))
        y_outer = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)[1] + int((dial_objects[-1].get_radius() + INTERMEDIATE_TICK_LEGTH) * np.sin(angle))

        cv.line(temp_display_frame, (x_inner, y_inner), (x_outer, y_outer), (255, 125, 0), LINE_STROKE_WEIGHT)
        cv.imshow('Calibration Window', temp_display_frame)

        # Show a zoomed in region
        zoomed_region = original_frame[y - 100 - HEADER_HEIGHT:y + 100 - HEADER_HEIGHT, x - 100:x + 100].copy()
        cv.line(zoomed_region, (100, 0), (100, 200), (0, 0, 255), 1)
        cv.line(zoomed_region, (0, 100), (200, 100), (0, 0, 255), 1)
        cv.imshow('Close-View', zoomed_region)
        
def set_threshold_intro(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        
        if len(dial_objects) <= 1: # first time running the calibration script. show the example threshold values infographic
            # Tell user to switch windows
            show_instructions("Switch to the other window to proceed...")
            cv.imshow('Calibration Window', display_frame)
            cv.setMouseCallback('Calibration Window', set_threshold)

            # Flag whether to show the example window
            global show_example_threshold
            show_example_threshold = True

            def example_thresholds_callback(event, x, y, flags, param):
                global show_example_threshold
                if event == cv.EVENT_LBUTTONDOWN:
                    show_example_threshold = False
                    cv.destroyWindow('Example thresholds')

            # Set the callback function for mouse events
            cv.namedWindow('Example thresholds')
            cv.setMouseCallback('Example thresholds', example_thresholds_callback)
            cv.imshow('Example thresholds', cv.imread(EXAMPLE_THRESHOLD_PATH))

            # Wait for a click before continuing
            while show_example_threshold:
                cv.waitKey(100)
        
        
        # Proceed to set threshold
        cv.imshow('Calibration Window', display_frame)
        
        # Create new display image for the individual dial
        threshold_display_image = np.ones((height + HEADER_HEIGHT, width, 3), dtype=np.uint8) * 255
        threshold_display_image[HEADER_HEIGHT:,:] = original_frame
        
        # Crop according to user's inputs
        ctr_coords = dial_objects[-1].get_headed_center_coords(HEADER_HEIGHT)
        avg_radius = dial_objects[-1].get_radius()
        cropped_dial = threshold_display_image[ctr_coords[1]-avg_radius:ctr_coords[1]+avg_radius, ctr_coords[0]-avg_radius:ctr_coords[0]+avg_radius]
        
        # Color correct and mask
        grey_dial = cv.cvtColor(cropped_dial, cv.COLOR_BGR2GRAY)
        inverted_dial = cv.bitwise_not(grey_dial)
        mask = np.zeros(inverted_dial.shape[:2], dtype='uint8')
        cv.circle(mask, (avg_radius, avg_radius), avg_radius, 255, -1)
        masked_dial = cv.bitwise_and(inverted_dial, inverted_dial, mask=mask)

        # Flag whether to show the example window
        global show_threshold_calibration
        show_threshold_calibration = True

        def threshold_calibration_callback(event, x, y, flags, param):
                global show_threshold_calibration
                if event == cv.EVENT_LBUTTONDOWN:
                    # Save the threshold value and kernel size to the dial
                    dial_objects[-1].threshold_value = cv.getTrackbarPos('Threshold Value', 'Threshold Calibration Window')
                    dial_objects[-1].kernel_size = cv.getTrackbarPos('Kernel Size', 'Threshold Calibration Window')
                    
                    show_threshold_calibration = False
                    cv.destroyWindow('Threshold Calibration Window')
                    show_instructions("Click anywhere to enter terminal values...")
                    cv.imshow('Calibration Window', display_frame)
                    cv.setMouseCallback('Calibration Window', enter_terminal_values)
        
        # Create the threshold calibration window and associated sliders
        cv.namedWindow('Threshold Calibration Window')
        
        # Set the default threshold value and kernel size based on the previous dial
        if len(dial_objects) > 1:
            threshold_value_default = dial_objects[-2].threshold_value
            kernel_size_default = dial_objects[-2].kernel_size
        else:
            threshold_value_default = 125
            kernel_size_default = 3

        cv.createTrackbar('Threshold Value', 'Threshold Calibration Window', threshold_value_default, 255, nothing)
        cv.createTrackbar('Kernel Size', 'Threshold Calibration Window', kernel_size_default, 10, nothing)
        cv.setMouseCallback('Threshold Calibration Window', threshold_calibration_callback)

        while show_threshold_calibration:
            thresholded_dial = cv.threshold(masked_dial, cv.getTrackbarPos('Threshold Value', 'Threshold Calibration Window'), 255, cv.THRESH_BINARY)[1]
            
            kernel = np.ones((int(cv.getTrackbarPos('Kernel Size', 'Threshold Calibration Window')), int(cv.getTrackbarPos('Kernel Size', 'Threshold Calibration Window'))), np.uint8)
            eroded_dial = cv.erode(thresholded_dial, kernel, iterations=1)
            dilated_dial = cv.dilate(eroded_dial, kernel, iterations=1)

            # Create new thresholded frame to display instructions on
            thresholded_frame = np.ones((dilated_dial.shape[1] + HEADER_HEIGHT, dilated_dial.shape[0], 3), dtype=np.uint8) * 255
            thresholded_frame[HEADER_HEIGHT:, :] = cv.cvtColor(dilated_dial, cv.COLOR_GRAY2BGR)

            # Determine font size and weight for threshold calibration window
            threshold_frame_message = "Click to save and continue"
            threshold_font_size = 10
            threshold_font_weight = 3
            while cv.getTextSize(threshold_frame_message, FONT, threshold_font_size, threshold_font_weight)[0][0] > 0.6 * thresholded_frame.shape[0]:
                threshold_font_size = threshold_font_size * 0.9
                threshold_font_weight = int(threshold_font_weight * 0.95)

            # Place text on the threshold calibration window
            threshold_textsize = cv.getTextSize(threshold_frame_message, FONT, threshold_font_size, threshold_font_weight)[0]
            cv.putText(thresholded_frame, threshold_frame_message, (thresholded_frame.shape[1]//2 - threshold_textsize[0]//2, HEADER_HEIGHT//2 + threshold_textsize[1]//2), cv.FONT_HERSHEY_SIMPLEX, threshold_font_size, (0, 0, 0), threshold_font_weight, cv.LINE_AA)

            cv.imshow('Threshold Calibration Window', thresholded_frame)
            cv.waitKey(1000)

def set_threshold(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        show_instructions("Wrong window! Proceed to the other window to set the threshold")
        cv.imshow('Calibration Window', display_frame)

def enter_terminal_values(event, x, y, flags, param):
     if event == cv.EVENT_LBUTTONDOWN:
        show_instructions("Please switch to the terminal to enter values...")
        cv.imshow('Calibration Window', display_frame)
        cv.waitKey(50)

         # get dial name from user
        dial_name = input("Enter the name of the dial: ")
        dial_objects[-1].name = str(dial_name)

        # get dial description from user
        dial_description = input("Enter a description of what the dial monitors: ")
        dial_objects[-1].description = str(dial_description)

        # get units from user
        units = input("Enter the units of the dial: ")
        dial_objects[-1].units = str(units)

            # get min value from user
        min_value = input("Enter the minimum value of the dial: ")
        dial_objects[-1].min_value = float(min_value)

        # get max value from user
        max_value = input("Enter the maximum value of the dial: ")
        dial_objects[-1].max_value = float(max_value)

        print("Return to the window to add more dials or finish calibration")
        print("")

        # proceed to draw center or finish calibration
        show_instructions("Click on the center of a new dial")
        show_button("Save and exit")
        cv.imshow('Calibration Window', display_frame)
        global dial_ident
        dial_ident = dial_objects[-1].ident + 1
        cv.setMouseCallback('Calibration Window', select_center)


# --- MAIN PROGRAM ---

cv.namedWindow('Calibration Window')
show_instructions("Click anywhere to begin calibration") # display intro text
cv.imshow('Calibration Window', display_frame)
cv.setMouseCallback('Calibration Window', intro) # set the callback function


# display the image
while True:
    #cv.imshow('Calibration Window', display_frame)
    key = cv.waitKey(100)