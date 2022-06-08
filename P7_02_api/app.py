# -*- coding: utf-8 -*-
import pickle
import json
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from functions.my_model_functions import *

app = Flask(__name__)

@app.route('/')
def home():
    #return "Hello!"
    return 'Ca marche!'

@app.route('/predict/<customer_id>', methods=['GET','POST'])
def predict(customer_id):

    customer_id = str(customer_id)

    # load database
    df_customers = pd.read_pickle("./database/df_customers_sample.pkl")

    # load model
    model = pickle.load(open('./database/model.pkl', 'rb'))

    # get customer details
    df_customer_id = customer_selection_function(df_customers, customer_id)

    # precit credit default
    prediction,  prediction_proba = predict_function(df_customer_id, model)

    # store results in a dict
    prediction_dict = {}
    prediction_dict.update({'customer_id': customer_id,
                            'prediction': prediction[0],
                            'prediction_proba': prediction_proba[0].max()
                           })

    return prediction_dict



if __name__ == '__main__':
    app.run()
