import matplotlib.pyplot as plot
import numpy as np
import cv2
import argparse


def convertimage(imagePath, detectFaceFlag):
    print(imagePath)
    print(detectFaceFlag)
    idexOfLastPathSeparator = imagePath.rindex('/', 0, len(imagePath))
    originalPath = imagePath[0:idexOfLastPathSeparator]
    fileName = imagePath[(idexOfLastPathSeparator + 1) : len(imagePath)]
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    if detectFaceFlag:
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(grayImage, 1.1, 4)
        if len(faces) < 1:
            count = 0
            while count < 4:
                # rotate image by 90 degree
                grayImage = rotateimage(grayImage, 90, count)
                faces = face_cascade.detectMultiScale(grayImage, 1.1, 4)
                if len(faces) >= 1:
                    break
                count += 1

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(grayImage, (x-30, y-50), (x+20 + w, y+40 + h), (0, 0, 0), -1)
        cv2.imwrite(originalPath + '/Enhanced_'+fileName+'.tif', grayImage)
    else:
        cv2.imwrite(originalPath + '/Enhanced_'+fileName+'.tif', grayImage)
        # Otsu's thresholding
        (ret2, th2) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(originalPath + '/Enhanced_otsu_'+fileName+'.tif', th2)


def rotateimage(image, angel, count):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    scale = 1.0
    print('rotating image')
    M = cv2.getRotationMatrix2D((cX, cY), angel, scale)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotatedimage = cv2.warpAffine(image, M, (nW, nH))
    return rotatedimage


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
