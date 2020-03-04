import matplotlib.pyplot as plot
import numpy as np
import cv2
import sys


def convertimage(imagePath, detectFaceFlag):
    print(imagePath)
    idexOfLastPathSeparator = imagePath.rindex('/', 0, len(imagePath))
    originalPath = imagePath[0:idexOfLastPathSeparator]
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    if detectFaceFlag:
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(grayImage, 1.1, 4)
        print(originalPath)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(grayImage, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.imwrite(originalPath + '/faceDetect_Dutch_passport.jpg', grayImage)
    else:
        cv2.imwrite(originalPath + '/GRAY_Dutch_passport.jpg', grayImage)

        # Otsu's thresholding
        (ret2, th2) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(originalPath + '/th2BW_Dutch_passport.jpg', th2)


if __name__ == '__main__':
    imagePath = (sys.argv[0])
    detectFaceFlag = bool(sys.argv[1])
    convertimage(imagePath, detectFaceFlag)
