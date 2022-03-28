import numpy as np
import cv2
from .configuration_running_redlight import RunningRedLightConfig


class RunningRedLight:
    def __init__(self, config: RunningRedLightConfig):
        self.config = config

    def update(self, detection, img):
        vehicle_violate = []

        #Detect traffic light state
        self.state = self.detect_state(img.copy())

        # For each object in detection, check intersection with stop line
        img_shape = img.shape
        if self.state == "red":
            for object in detection:
                bbox = object[0:4]
                poly_point = self.bbox_to_polygon(bbox, img_shape)
                violate = self.intersect_polygon_and_line(self.config.stop_line, poly_point)

                # if object violate add to list of vehicle violate
                if violate == True:
                    object_id = object[6]
                    vehicle_violate.append(object_id)
        
        return vehicle_violate


    def detect_state(self, img):
        # Get pixel size
        y_shape, x_shape, _ = img.shape

        # Convert polygon of traffic light area to pixel position
        points = [
            [int(x * x_shape), int(y * y_shape)]
            for x, y in self.config.traffic_light_area
        ]
        points = np.array(points)

        # Bounding rect
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        self.cropped = img[y : y + h, x : x + w].copy()

        # Create masking
        mask_points = points - points.min(axis=0)
        mask = np.zeros(self.cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [mask_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        tl_sumpx = np.sum(mask == 255)

        # Convert img from BGR to HSV
        img_hsv = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2HSV)

        # Color segmentation
        for color in ["red", "yellow", "green"]:
            hsv_min = np.array(getattr(self.config, f"{color}_hsv_min"))
            hsv_max = np.array(getattr(self.config, f"{color}_hsv_max"))

            img_seg = cv2.inRange(img_hsv, hsv_min, hsv_max)
            img_seg = cv2.bitwise_and(mask, img_seg)
            sum_px = np.sum(img_seg == 255)

            setattr(self, f"{color}_sumpx", sum_px)
            setattr(self, f"{color}_seg", img_seg)

        # Determine state
        for color in ["red", "yellow", "green"]:
            area_percent = int((getattr(self, f"{color}_sumpx") / tl_sumpx) * 100)
            threshold = getattr(self.config, f"{color}_min_area")
            setattr(self, f"{color}_area", area_percent)

            if area_percent >= threshold:
                return color

        return self.config.set_unknown_state

    def bbox_to_polygon(self, bbox, img_shape):
        ''' convert bbox to poly point'''

        # Get width and height img
        width = img_shape[1]
        height = img_shape[0]

        # Rescale bbox
        r_bbox = [0,0,0,0]
        r_bbox[0],r_bbox[2] = bbox[0]/width, bbox[2]/width
        r_bbox[1], r_bbox[3] = bbox[1]/height, bbox[3]/height


        # create point of polygon
        poly_point = []
        poly_point.append([r_bbox[0], r_bbox[1]])
        poly_point.append([r_bbox[0], r_bbox[3]])
        poly_point.append([r_bbox[2], r_bbox[1]])
        poly_point.append([r_bbox[2], r_bbox[3]])

        return poly_point
    
    def intersect_polygon_and_line(self, line, poly_point):
        ''' From list of poly point and line point return True if there is intersect between them'''
        line_x1, line_y1 = line[0]
        line_x2, line_y2 = line[1]

        i_next = 0
        for i_current in range(0, len(poly_point)):

            if i_current == len(poly_point) - 1:
                i_next = 0
            else:
                i_next = i_current + 1
            
            current_point = (poly_point[i_current][0], poly_point[i_current][1])
            next_point = (poly_point[i_next][0], poly_point[i_next][1])

            x3 = current_point[0]
            y3 = current_point[1]

            x4 = next_point[0]
            y4 = next_point[1]

            uA = ((x4 - x3) * (line_y1 - y3) - (y4 - y3) * (line_x1 - x3)) / (
                (y4 - y3) * (line_x2 - line_x1) - (x4 - x3) * (line_y2 - line_y1)
            )

            uB = ((line_x2 - line_x1) * (line_y1 - y3) - (line_y2 - line_y1) * (line_x1 - x3)) / (
                (y4 - y3) * (line_x2 - line_x1) - (x4 - x3) * (line_y2 - line_y1)
            )

            if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
                return True

        return False


    