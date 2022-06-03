from .main import app, api 
from flask_restful import Resource, reqparse
from flask import jsonify
from keras.models import Sequential
from skimage.transform import resize
import keras as k
import re
import numpy as np
import base64
import imageio
import tensorflow as tf

def init():
    with open("./modeling/result.json", "r") as rjson:
        loaded = rjson.read()
    model_load = tf.keras.models.model_from_json(loaded)
    model_load.load_weights("./modeling/model.h5")
    model_load.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
    graph = tf.compat.v1.get_default_graph
    return model_load, graph

def convert(img):
    with open('output.png','wb') as output:
	    output.write(base64.b64decode(img))

# model, graph = init()
# response = None

class ServeModel(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('URL', type=str, help="Please fill url image")

    def post(self):
        global response

        data = self.parser.parse_args()
        rjson = open("./modeling/result.json", "r")
        loaded = rjson.read()
        rjson.close()
        model_load = tf.keras.models.model_from_json(loaded)
        model_load.load_weights("./modeling/model.h5")
        model_load.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        graph = tf.compat.v1.get_default_graph()

        img_read = imageio.imread("./server/test.jpeg", pilmode="L")
        img_read = np.invert(img_read)
        img_read = resize(img_read, (28,28))
        img_read = img_read.reshape(1, 28, 28, 1)
        # print(img_read)
        # with graph.as_default():
        output = model_load.predict(img_read)
        response = np.array_str(np.argmax(output,axis=1))
            # model = Sequential()
            # output.call = tf.function(model.call)
        print(response)
        return jsonify({
            "output": str(output),
            "response": str(response)
        })