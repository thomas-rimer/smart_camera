import cv2 as cv
import math
import numpy as np

class Dial():
    SCAN_ANGLE_FIDELITY = 0.01
    WIPER_WIDTH = 3

    def __init__(self, ident, center_coords=None):

        self.ident = ident
        self.name = None
        self.center_coords = center_coords

        self.min_coords = None
        self.min_value = None

        self.max_coords = None
        self.max_value = None

        self.intermediate_tick_coords = []
        self.intermediate_tick_values = []

        self.units = None

        self.average_radius = None

        self.min_angle = None
        self.max_angle = None
        self.intermediate_tick_angles = []

        self.indicator_angle = None
        self.indicator_value = None
    
    def get_radius(self): # returns average distance between center and min/max points on circumference
        if not self.average_radius:
            dist_to_min = ((self.center_coords[0] - self.min_coords[0])**2 + (self.center_coords[1] - self.min_coords[1])**2)**0.5
            dist_to_max = ((self.center_coords[0] - self.max_coords[0])**2 + (self.center_coords[1] - self.max_coords[1])**2)**0.5
            self.average_radius = int((dist_to_min + dist_to_max)/2)
   
        return self.average_radius
    
    def get_headed_center_coords(self, header_height): # returns center coordinates offset to accomodate a display header 
        return self.center_coords[0], self.center_coords[1] + header_height

    def get_min_angle(self): # returns the angle of the min point relative to the center with the x-axis as 0 degrees
        if not self.min_angle:
            x_diff = self.min_coords[0] - self.center_coords[0]
            y_diff = self.center_coords[1] - self.min_coords[1]
            self.min_angle = math.atan2(y_diff, x_diff) % (2 * math.pi)

        return self.min_angle
    
    def get_max_angle(self): # returns the angle of the max point relative to the center with the x-axis as 0 degrees
        if not self.max_angle:
            x_diff = self.max_coords[0] - self.center_coords[0]
            y_diff = self.center_coords[1] - self.max_coords[1]
            self.max_angle = math.atan2(y_diff, x_diff) % (2 * math.pi)

        return self.max_angle

    def get_intermediate_tick_angles(self): # returns the angles of the intermediate tick points relative to the center with the x-axis as 0 degrees
        if len(self.intermediate_tick_angles) == 0:
            for tick in self.intermediate_tick_coords:
                x_diff = tick[0] - self.center_coords[0]
                y_diff = self.center_coords[1] - tick[1]
                self.intermediate_tick_angles.append(math.atan2(y_diff, x_diff) % (2 * math.pi))
        
        return self.intermediate_tick_angles
    
    def get_indicator_angle(self, frame): # returns the angle of the indicator relative to the center with the x-axis as 0 degrees

        wiper_value = 0 # arbitrary units
        wiper_angle = 0 # radians
        center_point = (self.get_radius(), self.get_radius())
        sweep_angle = 0
        while sweep_angle < 2 * math.pi: # Sweep a 'wiper' mask around the dial center to find the indicator
            # Calculate point on the circumference of the dial
            circumference_point = (int(center_point[0] + self.get_radius() * math.cos(sweep_angle)), int(center_point[1] - self.get_radius() * math.sin(sweep_angle)))
            
            # Mask the indicator with the wiper
            wiper_mask = np.zeros(frame.shape[:2], dtype='uint8')
            cv.line(wiper_mask, center_point, circumference_point, color=255, thickness=self.WIPER_WIDTH)
            pixels_under_wiper = cv.bitwise_and(frame, frame, mask=wiper_mask)

            # Calculate the wiper's value
            mean_wiper_value = cv.mean(pixels_under_wiper, mask=wiper_mask)[0]
            if mean_wiper_value > wiper_value:
                wiper_value = mean_wiper_value
                wiper_angle = sweep_angle
            
            sweep_angle += self.SCAN_ANGLE_FIDELITY
        
        if wiper_value == 0:
            raise ValueError("Indicator not detected after sweeping the entire dial.")
        
        # Save wiper angle as the indicator angle
        self.indicator_angle = wiper_angle

        return self.indicator_angle

    def get_indicator_value(self, frame): # returns the value of the indicator
        # Rotate all angles by -min_angle such that min_angle = 0
        adjusted_max_angle = (self.get_max_angle() - self.get_min_angle()) % (2 * math.pi)
        adjusted_intermediate_tick_angles = [(angle - self.get_min_angle()) % (2 * math.pi) for angle in self.get_intermediate_tick_angles()]
        adjusted_indicator_angle = (self.get_indicator_angle(frame) - self.get_min_angle()) % (2 * math.pi)

        # Flip such that dial always reads CCW
        if adjusted_intermediate_tick_angles[0] > adjusted_max_angle:
            adjusted_max_angle = 2 * math.pi - adjusted_max_angle
            adjusted_intermediate_tick_angles = [2 * math.pi - angle for angle in adjusted_intermediate_tick_angles]
            adjusted_indicator_angle = 2 * math.pi - adjusted_indicator_angle
        
        # Sort all tick marks along the dial's circumference
        all_ticks = list(zip([0] + adjusted_intermediate_tick_angles + [adjusted_max_angle], [self.min_value] + self.intermediate_tick_values + [self.max_value]))
        sorted_all_ticks = sorted(all_ticks, key=lambda x: x[0])
        all_tick_angles, all_tick_values = zip(*sorted_all_ticks)
        all_tick_angles = list(all_tick_angles)
        all_tick_values = list(all_tick_values)

        # Find the two tick marks that the indicator angle lies between
        index = 0
        indicator_value = None
        while index < len(all_tick_angles) - 1:
            if all_tick_angles[index] <= adjusted_indicator_angle <= all_tick_angles[index + 1]: # indicator angle is between these two tick marks
                # Interpolate the indicator value
                frac_into_section = (adjusted_indicator_angle - all_tick_angles[index]) / (all_tick_angles[index + 1] - all_tick_angles[index])
                indicator_value = all_tick_values[index] + frac_into_section * (all_tick_values[index + 1] - all_tick_values[index])
        
                self.indicator_value = indicator_value
                return self.indicator_value
            
            index += 1
        
        if not indicator_value:
            raise ValueError("Indicator is not within the dial's range.")