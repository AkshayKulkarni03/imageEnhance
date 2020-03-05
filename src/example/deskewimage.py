import matplotlib.pyplot as plot
import numpy as np
import cv2
import argparse

def deskew(image, angle):
    # convert the image to flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    image = cv2.bitwise_not(image)
    print('angle of deskew--' + str(angle))
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (cols, rows), center)

def compute_skew(image):
    # convert the image to flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.bitwise_not(image)
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=width / 2.0, maxLineGap=20)
    angle = 0.0
    no_of_lines = lines.size

    for x1, y1, x2, y2 in lines[0]:
        if x1 != x2:
            angle += np.arctan(y2 - y1 / x2 - x1)
    return angle / no_of_lines
