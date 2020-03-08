from typing import Dict, List

import numpy as np
from skimage.feature import canny
import cv2
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import pyplot as plt


def determine_skew(image):
    sigma = 3.0
    num_peaks = 20

    img = image
    edges = canny(img, sigma=sigma)
    # edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    plt.imshow(img, cmap='gray')
    plt.title('mser image'), plt.xticks([]), plt.yticks([])

    plt.show()

    out, angles, distances = hough_line(edges)
    _, angle_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=num_peaks)

    absolute_deviation = [determine_deviations(k) for k in angle_peaks]
    average_deviation = np.mean(np.rad2deg(absolute_deviation))
    angles_peaks_degree = [np.rad2deg(x) for x in angle_peaks]

    bin_0_45 = []
    bin_45_90 =[]
    bin_0_45n =[]
    bin_45_90n = []

    for angle in angles_peaks_degree:

        deviation_sum = int(90 - angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90.append(angle)
            continue

        deviation_sum = int(angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45.append(angle)
            continue

        deviation_sum = int(-angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45n.append(angle)
            continue

        deviation_sum = int(90 + angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90n.append(angle)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]

    nb_angles_max = 0
    max_angle_index = -1
    for angle_index, angle in enumerate(angles):
        nb_angles = len(angle)
        if nb_angles > nb_angles_max:
            nb_angles_max = nb_angles
            max_angle_index = angle_index

    if nb_angles_max:
        ans_arr = _get_max_freq_elem(angles[max_angle_index])
        angle = np.mean(ans_arr)
    elif angles_peaks_degree:
        ans_arr = _get_max_freq_elem(angles_peaks_degree)
        angle = np.mean(ans_arr)
    else:
        return None, angles, average_deviation, (out, angles, distances)

    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle

    return rot_angle, angles, average_deviation, (out, angles, distances)


def determine_deviations(angle):
    angle_in_degrees = np.abs(angle)
    deviation = np.abs(np.pi / 4 - angle_in_degrees)

    return deviation


def _get_max_freq_elem(peaks: List[int]):
    freqs: Dict[float, int] = {}
    for peak in peaks:
        if peak in freqs:
            freqs[peak] += 1
        else:
            freqs[peak] = 1

    sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
    max_freq = freqs[sorted_keys[0]]

    max_arr = []
    for sorted_key in sorted_keys:
        if freqs[sorted_key] == max_freq:
            max_arr.append(sorted_key)

    return max_arr


def _compare_sum(value) -> bool:
    return 44 <= value <= 46
