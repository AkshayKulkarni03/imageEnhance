import matplotlib.pyplot as plot
import numpy as np
import cv2
import argparse
from example.deskewimage import deskew
from deskew import determine_skew


# Convert image to gray scale and with otsu filter to get more cleat picture
# also with based on detect face flag face detection is triggered if its false directly
# deskwing code is invoked and tried to deskew the input image for test purposes all
# different images are stored for now
def convert_image(image_path, detect_face_flag):
    print(image_path)
    print(detect_face_flag)
    index_of_last_path_separator = image_path.rindex('/', 0, len(image_path))
    original_path = image_path[0:index_of_last_path_separator]
    file_name = image_path[(index_of_last_path_separator + 1): len(image_path)]
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if detect_face_flag:
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('src//example//haarcascade_frontalface_alt_tree.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
        if len(faces) < 1:
            count = 0
            while count < 4:
                # rotate image by 90 degree
                gray_image = rotate_image(gray_image, 90)
                faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
                if len(faces) >= 1:
                    break
                count += 1

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray_image, (x-30, y-50), (x+20 + w, y+40 + h), (0, 0, 0), -1)
        cv2.imwrite(original_path + '/Enhanced_'+file_name+'.tif', gray_image)
    else:
        cv2.imwrite(original_path + '/Enhanced_'+file_name+'.tif', gray_image)
        # determine the skew angle to rotate the image
        angle = determine_skew(gray_image)
        # deskew the image
        rotated = deskew(gray_image, angle, (255, 0, 0))
        cv2.imwrite(original_path + '/Enhanced_deskwed_' + file_name + '.tif', rotated)

        # Otsu's thresholding
        (ret2, th2) = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(original_path + '/Enhanced_otsu_'+file_name+'.tif', th2)


def rotate_image(image, angel):
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

    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    return rotated_image


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
    parser = argparse.ArgumentParser(description= 'Enhance the Image and convert to tiff.')
    parser.add_argument('image', metavar='image_path', type=str, help='An input image path to enhance the image')
    parser.add_argument('detect_face', metavar='detect_face', type=bool, help='Detect Face Flag true/false')
    args = parser.parse_args()
    print(args)
    image_path = (sys.argv[1])
    detect_face_flag = str2bool(sys.argv[2])
    convert_image(image_path, detect_face_flag)
