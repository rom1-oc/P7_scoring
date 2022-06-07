import pickle
#import shap
import lime
import requests
import urllib.request, json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
from functions.dashboard_functions import *

# load dataframes
df_customers = pickle.load(open('./database/df_customers_sample.pkl', 'rb'))
df_radar = pickle.load(open('./database/df_radar_sample.pkl', 'rb'))

# load model
model = pickle.load(open('./database/model.pkl', 'rb'))

#import .css style sheet
with open('static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def display_function(data, customer_id):

    df_customers['DAYS_BIRTH_YEARS'] = -df_customers['DAYS_BIRTH']/365
    df_customers['DAYS_EMPLOYED_YEARS'] = -df_customers['DAYS_EMPLOYED']/365

    df_customer_id = get_customer_data(df_customers, customer_id)
    df_customer_id_radar = get_customer_data(df_radar, customer_id)

    st.write(f"<style color=red>XXX</style>", unsafe_allow_html=True)
    #
    st.header("")
    st.subheader("Credit Decision:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customer", data["customer_id"])
    col2.metric("Status", "DENIED" if data["prediction"] == 1 else "APPROVED", )
    col3.metric("Confidence (%)", round(float(data["prediction_proba"]*100), 2))

    #
    st.header("")
    st.subheader("Customer Profil:")
    radar_features = ['AMT_INCOME_TOTAL',
                      'AMT_CREDIT',
                      'CNT_CHILDREN',
                      'AMT_GOODS_PRICE',
                      'DAYS_EMPLOYED_YEARS']

    customer_profil_function(df_customer_id_radar, customer_id, radar_features)

    #
    st.header("")
    st.subheader("Customer Features:")
    st.dataframe(df_customer_id)

    with st.expander("Details"):
        #
        # st.header("")
        # st.subheader("Local Feature Importance")
        # feature_importance_local_function(df_customers,
        #                                   df_customer_id,
        #                                   model)

        #
        st.header("")
        st.subheader("Global Feature Importance")
        feature_importance_global_plotly_function(df_customer_id.columns.tolist(),
                                             model,
                                             len(df_customer_id.columns.tolist())
                                             )

    #
    st.header("")
    st.subheader("Pie Features Repartition:")

    #
    pie_group = st.selectbox('Select group:', ['TARGET'])

    #
    pie_cat_features = ['OCCUPATION_TYPE',
                        'NAME_CONTRACT_TYPE',
                        'NAME_INCOME_TYPE',
                        'NAME_EDUCATION_TYPE',
                        'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE']

    options_pie = st.multiselect('Select category(ies):',
                                 pie_cat_features,
                                 [])


    if len(options_pie) > 0:
        graph_pie_plotly_function(options_pie, df_customers, df_customer_id, pie_group)

    #
    st.header("")
    st.subheader("Hist Bar Customer Position:")

    #
    hist_bar_group = st.selectbox('Select group:     ', ['TARGET'])

    #
    hist_bar_cat_features = ['AMT_CREDIT',
                             'AMT_INCOME_TOTAL',
                             'DAYS_BIRTH_YEARS',
                             'DAYS_EMPLOYED_YEARS',
                             'AMT_ANNUITY',
                             'AMT_GOODS_PRICE',
                              #'AGE_GROUP_DEFAULT',
                              #'OCCUPATION_TYPE',
                             ]

    #
    options_hist_bar = st.multiselect('Select category(ies):',
                                      hist_bar_cat_features,
                                      [])

    #
    if len(options_hist_bar) > 0 :

        graph_hist_bar_plotly_function(options_hist_bar,
                                       df_customers,
                                       df_customer_id,
                                       hist_bar_group)
        #graph_hist_bar_function(options_hist_bar[0], df_customers, df_customer_id)


    #
    st.header("")
    st.subheader("Box Customer Position:")

    #
    box_group = st.selectbox('Select group: ', ['TARGET'])

    #
    options_box_cat = st.selectbox('Select category:',
                                   ('-',
                                   'OCCUPATION_TYPE',
                                   'NAME_CONTRACT_TYPE',
                                   'NAME_INCOME_TYPE',
                                   'NAME_EDUCATION_TYPE',
                                   'NAME_FAMILY_STATUS',
                                   'NAME_HOUSING_TYPE'))

    #
    options_box_num = st.selectbox('Select metric:',
                                   ('-',
                                   'AMT_INCOME_TOTAL',
                                   'AMT_CREDIT',
                                   'DAYS_BIRTH_YEARS',
                                   'DAYS_EMPLOYED_YEARS'))


    if options_box_cat != '-' and options_box_num != '-':

        #
        graph_box_plotly_function(options_box_cat,
                           options_box_num,
                           df_customers,
                           df_customer_id,
                           box_group)

    #
    st.header("")
    st.subheader("Scatter Customer Position:")

    #
    scatter_group = st.selectbox(' Select group:', ['TARGET'])

    #
    options_scatter_num_1 = st.selectbox('Select 1st metric:',
                                         ('-',
                                          'AMT_INCOME_TOTAL',
                                          'AMT_CREDIT',
                                          'DAYS_BIRTH_YEARS',
                                          'DAYS_EMPLOYED_YEARS'))
    #
    options_scatter_num_2 = st.selectbox('Select 2nd metric:',
                                         ('-',
                                          'AMT_INCOME_TOTAL',
                                          'AMT_CREDIT',
                                          'DAYS_BIRTH_YEARS',
                                          'DAYS_EMPLOYED_YEARS'))

    if options_scatter_num_1 != '-' and options_scatter_num_2 != '-' :

        #
        graph_scatter_plotly_function(options_scatter_num_1,
                                       options_scatter_num_2,
                                       df_customers,
                                       df_customer_id,
                                       scatter_group)

    return

def main():

    st.title('Credit Default Dashboard')

    customer_id = st.selectbox('Enter customer ID:',
                               (['-'] + list(df_customers.index)))

    if customer_id.isdigit():
        # get json data from Flask API
        with urllib.request.urlopen(f"http://127.0.0.1:5000/predict/{customer_id}") as url:
        #with urllib.request.urlopen(f"https://final-app-p7.herokuapp.com/predict/{customer_id}") as url:
            data = json.loads(url.read())

        display_function(data, customer_id)



#
main()
