from .main import app, api 
from flask_restful import Resource, reqparse
from flask import jsonify
from skimage.transform import resize
import numpy as np
import imageio
import tensorflow as tf
import keras as K
from keras.models import load_model, model_from_json
import pandas as pd

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.backend.get_session().run(tf.local_variables_initializer())
    return auc

model_load = load_model('./modeling/model_repfit_v1.h5', custom_objects={'rmse': rmse})
model_load.load_weights("./modeling/model_repfit_weight_v1.h5")

class ServeModelTF(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('product_score')
        self.parser.add_argument('qty')
        self.parser.add_argument('freight_price')
        self.parser.add_argument('product_weight_g')
        self.parser.add_argument('lag_price')
        self.parser.add_argument('comp1')
        self.parser.add_argument('ps1')
        self.parser.add_argument('fp1')
        self.parser.add_argument('comp2')
        self.parser.add_argument('ps2')
        self.parser.add_argument('fp2')
        self.parser.add_argument('bed_bath_table')
        self.parser.add_argument('computers_accessories')
        self.parser.add_argument('consoles_games')
        self.parser.add_argument('cool_stuff"')
        self.parser.add_argument('furniture_decor')
        self.parser.add_argument('garden_tools')
        self.parser.add_argument('health_beauty')
        self.parser.add_argument('perfumery')
        self.parser.add_argument('watches_gifts')

    def post(self):

        data = self.parser.parse_args()
        print(data["fp1"])

        x=pd.DataFrame.from_dict(data, orient='index').transpose()

        output = str(model_load.predict(x)[0][0])
        print(output)
        data['prediction'] = output

        return jsonify(data)
        