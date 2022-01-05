import cv2

"""
Author: Ante Zovko
Date: Jan 4th, 2022
Description: Given an image, this program will find the contours of the image and
export the individual detected contours as images

"""


def start(image_path='../Picture_Upload_Service/Image/expression.jpg'):
    # Import image
    img = cv2.imread(image_path)

    # Read as grayscale,(Creates CV_8UC1 single channel image and the bounding rectangles can only be
    # grayscale)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours test
    # cv2.drawContours(Image, contours, -1, (0, 255, 255), 3)

    # Iterate through contours for sorting
    image_counter = 0
    # Create a list of contours
    contour_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Eliminate noise by checking if contour is large enough
        if w > 40 and h > 15:
            contour_list.append([x, y, w, h])

    # Sort contours by x position
    contour_list.sort(key=lambda x: x[0])

    for contour in contour_list:
        x, y, w, h = contour
        # Draw bounding rectangles
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Save image
        if image_counter == 0:
            pass
        else:
            cv2.imwrite('../Detected_Images/' + str(image_counter) + '.jpg', threshold[y:y + h, x:x + w])
        image_counter += 1


if __name__ == '__main__':
    start()

# Display countours
# cv2.drawContours(Image, contours, -1, (0, 255, 255), 3)
