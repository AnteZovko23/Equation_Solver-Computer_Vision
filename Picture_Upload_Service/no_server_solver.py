import sys
import argparse

sys.path.insert(0, '')
import server



"""
Author: Ante Zovko
Date: Jan 3rd, 2022
Description: Performs classification on a given image without a server running

"""

# Run cmd

# An image path command line argument using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='./Image_demo/Demo.jpeg', help='Path to image')
args = parser.parse_args()

if __name__ == '__main__':
    server.start_recognition(args.image_path)


