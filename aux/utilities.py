import numpy as np
import math


def points_of_contours(contours):
    # Get all the points from contours and return an array
    points = []
    for contour in contours:
        for point in contour:
            points.append(point)
    points = np.asarray(points)

    return points


def euclidean_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cheat_basic(x_distance, w):
    if x_distance > w / 2:
        action = 'right'
    elif x_distance < -w / 2:
        action = 'left'
    else:
        action = 'attack'

    return action
