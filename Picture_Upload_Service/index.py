import sys
import os

sys.path.insert(0, '..')
from src import Contour_Recognition
from Tensorflow import Classification

import tensorflow as tf

import tornado.web
import tornado.ioloop

"""
Author: Ante Zovko
Date: Jan 3rd, 2022
Description: An image uploader for the Tensorflow classifier

"""


class UploadImgHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["imgFile"]
        for f in files:
            fh = open(f"./Image/expression.jpg", "wb")
            fh.write(f.body)
            fh.close()
        self.write(f"Success!")
        expression = start_recognition().split(" ")
        self.write("<br>")
        self.write("<br>")
        self.write("<br>")
        self.write("<br>")
        self.write(expression[0])
        self.write("<br>")
        self.write("<br>")
        self.write(expression[1] + "" + expression[2])

    def get(self):
        self.render("index.html")


def start_recognition(image_path='../Picture_Upload_Service/Image/expression.jpg',
                      model_path='../Tensorflow/saved_model_2/saved_model/my_model'):
    # delete all files in the Image folder
    for file in os.listdir("../Detected_Images"):
        os.remove(os.path.join("../Detected_Images", file))

    Contour_Recognition.start(image_path)

    return Classification.classify(tf.keras.models.load_model(model_path))


def start_server():
    app = tornado.web.Application([
        ("/", UploadImgHandler),
        ("/Image/(.*)", tornado.web.StaticFileHandler, {'path': 'Image'})
    ])

    app.listen(8080)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    start_server()
