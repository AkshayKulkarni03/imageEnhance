import os
import argparse
import sys
from src.example.enhanceImage import convert_image, str2bool
from src.example import enhanceImage


def main():
    """Entry point for the application script"""
    print("Call image enhancer")
    parser = argparse.ArgumentParser(description='Enhance the Image and convert to tiff.')
    parser.add_argument('image', metavar='image_path', type=str, help='An input image path to enhance the image')
    parser.add_argument('detect_face', metavar='detect_face', type=bool, help='Detect Face Flag true/false')
    args = parser.parse_args()
    print(args)
    image_path = (sys.argv[1])
    detect_face_flag = str2bool(sys.argv[2])
    convert_image(image_path, detect_face_flag)