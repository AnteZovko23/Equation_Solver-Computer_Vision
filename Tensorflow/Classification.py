"""
Author: Ante Zovko
Date: Jan 3rd, 2022
Description: Testing Convolutional Neural Network that recognizes handwritten mathematical symbols
[(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','/', 'x']

"""
CUDA_VISIBLE_DEVICES = ""
import numpy as np
import os
import cv2


def classify(model):
    # load model

    # labels
    label_dictionary = {
        0: "(",
        1: ")",
        2: "+",
        3: "-",
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        10: 6,
        11: 7,
        12: 8,
        13: 9,
        14: "/",
        15: "x"
    }

    # Evaluates a given expression
    def evaluate_expression(input_string):
        first_number = ""
        result = 0
        start = 0
        # start with first number
        for i in range(len(input_string)):
            if input_string[i] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                first_number += str(input_string[i])

            else:
                start = i
                result = int(first_number)
                break

        # iterate through the rest of the list
        # if the current character is a number, add it to the result
        # if the current character is an operator, perform the operation
        for character in range(start, len(input_string)):
            if input_string[character] in ['+', '-', 'x', '/']:
                if input_string[character + 1] in ['+', '-', 'x', '/']:
                    continue
                next_value = get_next_number(input_string[character + 1: len(input_string)])
                if input_string[character] == '+':
                    result += next_value

                elif input_string[character] == '-':
                    result -= next_value

                elif input_string[character] == 'x':
                    result *= next_value
                elif input_string[character] == '/':
                    result /= next_value

            elif input_string[character] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                continue

            else:
                result = int(input_string[character])

        return result

    image_list = []
    # Load in the images
    for filepath in sorted(os.listdir('../Detected_Images/')):
        given_image = cv2.imread('../Detected_Images/{0}'.format(filepath), 0)
        resized_image = cv2.resize(given_image, (45, 45))
        image_array = np.expand_dims(resized_image, axis=0)
        image_list.append(image_array)

    expression = []
    evaluated_expression = ""
    # Classify the images
    for image in image_list:
        prediction = model.predict(image)
        expression.append(label_dictionary[np.argmax(prediction)])
        print(label_dictionary[np.argmax(prediction)], end="")
        evaluated_expression += str(label_dictionary[np.argmax(prediction)])

    print("\n")

    print("Result: {}".format(evaluate_expression(expression)))
    evaluated_expression += " Result: {}".format(evaluate_expression(expression))

    return evaluated_expression


def get_next_number(input_string):
    # get the next number in the expression
    next_number = ""

    # start with first number
    for i in range(len(input_string)):
        if input_string[i] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            next_number += str(input_string[i])
        else:
            break

    return int(next_number)


if __name__ == '__main__':
    pass
