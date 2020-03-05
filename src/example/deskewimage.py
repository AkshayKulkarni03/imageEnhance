from typing import Tuple, Union
import numpy as np
import cv2


# Deskew the image based on inputs mentioned as parameters
def deskew(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    # load the height and width from shape of image
    old_width, old_height = image.shape[:2]

    # convert the angle in degrees to radian to get exact rotations
    angle_radian = np.math.radians(angle)

    # calculate the new width and height based on absolute values of sin and cos of original width and height
    # of image with proper calculation to get full image alignment covered
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    # reversed the first 2 items of an original image shape and then slicing array into half and getting new tuple list
    # to load the image center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    # Map the 2d matrix of an image with center alignment and with determined new angle
    # and update the particular elements of an array with respect to height and width
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    # Rotate the image with rounded height, width and with above plotted matrix
    # and setting the remaining portion of image with color as background color passed to an method
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
