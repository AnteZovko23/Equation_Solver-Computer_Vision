import sys
import argparse
import os
from random import choice

sys.path.insert(0, '')
import app

"""
Author: Ante Zovko
Date: Jan 3rd, 2022
Description: Performs classification on a given image without a server running

"""

# Run cmd

# An image path command line argument using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='', help='Path to image')
args = parser.parse_args()
image = ""
# If image path argument is empty select random image from Image Demo folder
if args.image_path == '':
    image = './Image_demo/' + choice(os.listdir('Image_demo/'))
else:
    image = args.image_path

if __name__ == '__main__':
    print("\n")
    print("Image selected: " + image)
    print("\n")
    app.start_recognition(image)
