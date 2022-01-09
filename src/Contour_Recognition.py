import cv2
import numpy as np
from skimage.draw import polygon_perimeter

"""
Author: Ante Zovko
Date: Jan 4th, 2022
Description: Given an image, this program will find the contours of the image and
export the individual detected contours as images

"""


def start(image_path):
    # Import image

    img = cv2.imread(image_path, -1)

    rgb_planes = cv2.split(img)
    # Eliminate shadows
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    img = cv2.merge(result_norm_planes)
    # # Save image on disk
    # img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # # Apply gaussian blur, threshold, and erode
    # Apply threshold
    img = cv2.GaussianBlur(img, (7, 7), 0)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # img = cv2.bitwise_not(img)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    cv2.imwrite("Test.jpeg", img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iterate through contours for sorting
    image_counter = 0
    # Create a list of contours
    contour_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Eliminate noise by checking if contour is large enough
        if w > 50 and h > 10:
            contour_list.append([x, y, w, h])

    # Sort contours by x position
    contour_list.sort(key=lambda x: x[0])

    # computes the bounding box for the contour, and draws it on the frame,
    for contour in contour_list:
        # if w > 10 and h > 10:
        x, y, w, h = contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save image
        if image_counter == 0:
            pass
        else:

            cropped_img = img[y:y + h, x:x + w]

            cropped_img = cv2.dilate(cropped_img, kernel, iterations=2)

            cv2.imwrite('./Detected_Images/' + str(image_counter) + '.png', cropped_img)
        image_counter += 1


    # img = cv2.imwrite("Test.jpeg", img)

# Display countours
# cv2.drawContours(Image, contours, -1, (0, 255, 255), 3)
