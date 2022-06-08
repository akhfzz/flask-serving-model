from .main import app, api 
from flask_restful import Resource, reqparse
from flask import jsonify
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from .predict import rmse


model = load_model('./modeling/model_repfit_v1.h5', custom_objects={'rmse': rmse})
model.load_weights('./modeling/model_repfit_v1_weights.h5')

class ServeModelTF(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('product_score', type=float)
        self.parser.add_argument('qty', type=float)
        self.parser.add_argument('freight_price', type=float)
        self.parser.add_argument('product_weight_g', type=float)
        self.parser.add_argument('lag_price', type=float)
        self.parser.add_argument('comp1', type=float)
        self.parser.add_argument('ps1',type=float)
        self.parser.add_argument('fp1', type=float)
        self.parser.add_argument('comp2', type=float)
        self.parser.add_argument('ps2', type=float)
        self.parser.add_argument('fp2', type=float)
        self.parser.add_argument('bed_bath_table', type=float)
        self.parser.add_argument('computers_accessories', type=float)
        self.parser.add_argument('consoles_games',type=float)
        self.parser.add_argument('cool_stuff"', type=float)
        self.parser.add_argument('furniture_decor', type=float)
        self.parser.add_argument('garden_tools', type=float)
        self.parser.add_argument('health_beauty', type=float)
        self.parser.add_argument('perfumery', type=float)
        self.parser.add_argument('watches_gifts', type=float)

    def post(self):

        data = self.parser.parse_args()

        if (data != None):
            x=pd.DataFrame.from_dict(data, orient='index').transpose()
            print(x)
    
        prediction_result = str(model.predict(x)[0][0])
        price_predict = (float(prediction_result) * (364.900000 - 19.900000) + 19.900000)

        return jsonify({
            "status": True,
            "price": price_predict,
            "predict": prediction_result
        })
        