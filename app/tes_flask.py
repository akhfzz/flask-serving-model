# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras as K
from keras.models import load_model, model_from_json

# instantiate flask 
app = flask.Flask(__name__)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph
graph = tf.compat.v1.get_default_graph()
model = load_model('./modeling/model_repfit_v1.h5', custom_objects={'rmse': rmse})
model.load_weights('./modeling/model_repfit_weight_v1.h5')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        print(x)
        # with graph.as_default():
        data["prediction"] = str(model.predict(x)[0][0])
        data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')