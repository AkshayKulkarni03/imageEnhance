import matplotlib.pyplot as plot
import numpy as np
import cv2
import argparse


def convertimage(imagePath, detectFaceFlag):
    print(imagePath)
    print(detectFaceFlag)
    idexOfLastPathSeparator = imagePath.rindex('/', 0, len(imagePath))
    originalPath = imagePath[0:idexOfLastPathSeparator]
    fileName = imagePath[idexOfLastPathSeparator +1 : len(imagePath)]
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    if detectFaceFlag:
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(grayImage, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(grayImage, (x-30, y-50), (x+20 + w, y+40 + h), (0, 0, 0), -1)
        cv2.imwrite(originalPath + '/Enhanced_'+fileName+'.tif', grayImage)
    else:
        cv2.imwrite(originalPath + '/Enhanced_'+fileName+'.tif', grayImage)
        # Otsu's thresholding
        (ret2, th2) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(originalPath + '/Enhanced_otsu_'+fileName+'.tif', th2)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    import sys
    imagePath = (sys.argv[1])
    detectFaceFlag = str2bool(sys.argv[2])
    convertimage(imagePath, detectFaceFlag)
