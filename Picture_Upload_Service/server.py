import base64
import numpy as np
import io
import sys
import os

sys.path.insert(0, '../')
from src import Contour_Recognition
from Tensorflow import Classification

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2

"""
Author: Ante Zovko
Date: Jan 8rd, 2022
Backend server with Flask and Tensorflow

"""

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/home')
def home():
    return render_template('index.html')


@app.route("/process", methods=["POST"])
def post():
    # get the image from the request
    message = request.get_json(force=True)
    encoded_image = message['image']
    decoded_image = base64.b64decode(encoded_image)
    image_data = io.BytesIO(decoded_image)

    # convert the image to a numpy array
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize the image to a smaller size
    image = cv2.resize(image, (1920, 1080))
    # Save
    cv2.imwrite("./Image/image.jpeg", image)

    expression = start_recognition("./Image/image.jpeg").split(" ")

    response = {

        "expression": expression[0],
        "result": expression[2]

    }

    return jsonify(response)


def start_recognition(image_path, model_path='./templates/saved_model_2/saved_model/my_model'):
    # delete all files in the Image folder
    for file in os.listdir("../Detected_Images"):
        os.remove(os.path.join("../Detected_Images", file))

    Contour_Recognition.start(image_path)

    return Classification.classify(tf.keras.models.load_model(model_path))
