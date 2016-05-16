import cv2
import numpy as np
import aux.utilities as utilities


class FeatureExtractor:
    # Extract features from current state
    def getFeatures(self, state, action):
        raise Exception('Not implemented function')


class BasicMapExtractor(FeatureExtractor):
    # Extract:
    #        -distance (x axis) to target
    #        -target size
    #        -target position in screen
    def getFeatures(self, state, action=None, ret_img=False):
        features = {}
        img = state.image_buffer

        blueLower = np.array([0, 0, 0], dtype='uint8')
        blueUpper = np.array([255, 0, 0], dtype='uint8')

        blue = cv2.inRange(img, blueLower, blueUpper)

        (_, contours, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = utilities.points_of_contours(contours)
        (x, y, w, h) = cv2.boundingRect(points)
        rectangle_center = (w / 2 + x, h / 2 + y)

        mid = (img.shape[1] // 2, img.shape[0] // 2)
        distance = rectangle_center[0] - mid[0]

        features['closer'] = 0.0
        features['in_target'] = 0.0
        if distance < w/4 and action == 'right':
            features['closer'] = 1.0
        elif distance > w/4 and action == 'left':
            features['closer'] = 1.0
        elif abs(distance) > w and action == 'attack':
            features['in_target'] = 1.0






        if ret_img:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(img, rectangle_center, 3, (0, 2550, 0))
            return [img, blue], features
        else:
            return features
