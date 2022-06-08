import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer


def predict_function(config, model):

    if type(config) == dict:
        dataframe = pd.DataFrame(config)
    else:
        dataframe = config

    y_pred = model.predict(dataframe)
    y_pred_proba = model.predict_proba(dataframe)

    return y_pred, y_pred_proba


def customer_selection_function(dataframe, customer_id):

    dataframe = dataframe.filter(items=[customer_id], axis=0)

    return dataframe
