import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lime import lime_tabular


# load model
model = pickle.load(open('./database/model.pkl', 'rb'))

def get_customer_data(df_customers, customer_id):
    """
    Gets customer data by filtering rows

    Args:
        df_customers (pd.Dataframe): data source
        customer_id (int):
    Returns:
        df_customer_id (pd.Dataframe): data output

    """
    df_customer_id = df_customers.copy()
    df_customer_id = df_customer_id.filter(items=[customer_id], axis=0)
    #df_customer_id = df_customer_id.drop(columns={"TARGET"})

    return df_customer_id


def inter_quartile_method_function(dataframe):
    """
    Apply inter-quartile method on each column to remove outilers

    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        dataframe (pd.Dataframe): data output
    """
    #
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1

    #
    dataframe = dataframe[(dataframe <= dataframe.quantile(0.75) + 1.5*iqr)
                                                        & (dataframe >= dataframe.quantile(0.25) - 1.5*iqr)]

    #
    return dataframe


def feature_importance_local_function(df_customers, df_customer_id, model):
    """
    Displays LIME plot (local feature importance)

    Args:
        params (dict): model parameters
        predictions (boolean):
    Returns
        -
    """
    #
    # n_features = 158
    #
    # explainer = lime_tabular.LimeTabularExplainer((model.named_steps['pre-processor'].transform(df_customers))[:,:n_features],
    #                                               mode="classification",
    #                                               class_names=np.unique(df_customers['TARGET']),
    #                                               feature_names=df_customers.columns.tolist()
    #                                              )
    #
    # data_customer_id = (model.named_steps['pre-processor'].transform(df_customer_id))[:,:n_features]
    # data_customer_id = data_customer_id.flatten()
    #
    # st.table(data_customer_id)
    # st.write(model.predict_proba)
    #
    # explanation = explainer.explain_instance(data_customer_id,
    #                                          model.predict_proba,
    #                                          num_features=n_features)
    #
    # components.html(explanation.as_html(), height=800)
    # explanation.as_list()

    return

def feature_importance_global_plotly_function(labels, model, n_features):
    """
    Displays global feature importances plot

    Args:
        labels list(str): features name
        model (.pkl): notebook model
        n_features (int): number of features
    Returns
        -
    """
    #
    importance = model.named_steps['classifier'].coef_[0]
    importance = importance[:n_features]

    # associate labels and importance score
    my_dict = {labels[i]: importance[i] for i in range(len(labels))}

    # sort dict by importance score
    my_list = sorted(my_dict.items(), key=lambda x:x[1], reverse=True)
    my_dict = dict(my_list)

    df = pd.DataFrame(data=my_dict.values(), index=my_dict.keys())
    df['Contribution (%)'] = df[0]
    df = df.reset_index()
    df = df.rename(columns={'index':'Features'})

    fig = px.bar(df,
                 x='Features',
                 y='Contribution (%)',
                 title=('A.I. Algorithm:'))

    fig.update_layout(autosize=False,
                      #width=500,
                      height=600,
                      )



    st.plotly_chart(fig, use_container_width=True)


    return


def graph_pie_plotly_function(features, df_customers, df_customer_id, pie_group):
    """
    Displays dynamic grid of pie plots

    Args:
        features list(str): selected features by end-user
        df_customers (pd.DataFrame): database
        df_customer_id (pd.DataFrame): customer data
        target (str): selected target ("APPROVED" or "DENIED") by end-user
    Returns
        -
    """

    df_customers_0 = df_customers[df_customers[pie_group] == 0]
    df_customers_1 = df_customers[df_customers[pie_group] == 1]

    # Subplot
    #int(math.ceil(len(features)/2))
    nrows = 2
    ncols = 2

    #Create subplots, using 'domain' type for pie charts
    specs = [[{'type':'domain'}, {'type':'domain'}],
             [{'type':'domain'}, {'type':'domain'}],
            ]

    fig = make_subplots(rows=nrows,
                        cols=ncols,
                        specs=specs,
                        subplot_titles=['Status: ACCEPTED', 'Status: DENIED']
                        )

    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        if i < len(features) :

            feature = features[i]
            customer_label = df_customer_id[feature].values[0]


            labels_0 = df_customers_0[feature].value_counts().index
            labels_1 = df_customers_1[feature].value_counts().index
            values_0 = df_customers_0[feature].value_counts().values
            values_1 = df_customers_1[feature].value_counts().values


            my_i_0 = list(df_customers_0[feature].value_counts().index).index(customer_label)
            my_pull_0 = [0] * len(labels_0)
            my_pull_0[my_i_0] = 0.15

            my_i_1 = list(df_customers_1[feature].value_counts().index).index(customer_label)
            my_pull_1 = [0] * len(labels_1)
            my_pull_1[my_i_1] = 0.15

            fig.add_trace(go.Pie(labels=labels_0,
                                 values=values_0,
                                 name="xxx",
                                 textinfo='percent+label',
                                 title=feature,
                                 hole=0.8,
                                 pull=my_pull_0,
                                 #legendgroup = str(nrows_ + 1)
                                 ),
                                 nrows_ + 1,
                                 ncols_ + 1)

            fig.add_trace(go.Pie(labels=labels_1,
                                 values=values_1,
                                 name="xxx",
                                 textinfo='percent+label',
                                 title=feature,
                                 hole=0.8,
                                 pull=my_pull_1,
                                 #legendgroup =str(nrows_ + 1)
                                 ),
                                 nrows_ + 1,
                                 ncols_ + 2)



            fig.update_traces(showlegend=False,
                             )


            # parameters
            fig.update_layout(autosize=False,
                              width=500,
                              height=1000,
                              )

        i = i + 1
    nrows_ = nrows_ + 1

    st.plotly_chart(fig, use_container_width=True)

    return

def graph_hist_bar_plotly_function(features, df_customers, df_customer_id, hist_bar_group):
    """
    Displays dynamic grid of hist and bar plots

    Args:
        features list(str): selected features by end-user
        binary_feature (str): selected binary feature by end-user
        df_customers (pd.DataFrame): database
        df_customer_id (pd.DataFrame): customer data
        target (str): selected target ("APPROVED" or "DENIED") by end-user
    Returns
        -
    """

    df_customers_0 = df_customers[df_customers[hist_bar_group] == 0]
    df_customers_1 = df_customers[df_customers[hist_bar_group] == 1]

#     feature = features[0]

#     if feature == "OCCUPATION_TYPE":
#
#         #
#         fig = plt.figure(figsize=(12,8))
#         ax = fig.subplots()
#
#         df_customers_plot = (df_customers.groupby('OCCUPATION_TYPE')
#                                          .agg({'OCCUPATION_TYPE':'count'})
#                                          .rename(columns={'OCCUPATION_TYPE':'OCCUPATION_TYPE_COUNT'})
#                                          .sort_values(by=['OCCUPATION_TYPE_COUNT'], ascending=False)
#                             )
#
#         customer_id_occupation = df_customer_id['OCCUPATION_TYPE'].values[0]
#         df_n_bin = df_customers_plot.reset_index()[df_customers_plot.reset_index()['OCCUPATION_TYPE'] == customer_id_occupation].index
#         n_bin = df_n_bin[0]
#
#         bars = ax.bar(df_customers_plot.index.astype(str),
#                       df_customers_plot['OCCUPATION_TYPE_COUNT'],
#                       edgecolor = 'k')
#
#         bars[n_bin].set_color('r')
#
#         ax.tick_params(axis='x', rotation=45)
#         ax.set_xlabel('Occupation Type')
#         ax.set_ylabel('Count')
#         ax.set_title('Number of loans by occupation')
#         st.pyplot(fig)
#
#
#     elif feature == "AGE_GROUP_DEFAULT":
#
#         # Age information into a separate dataframe
#         age_data = df_customers[['TARGET', 'DAYS_BIRTH']]
#         age_data['DAYS_BIRTH'] = -age_data['DAYS_BIRTH']
#         age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
#         # Bin the age data
#         age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
#         # Group by the bin and calculate averages
#         age_groups  = age_data.groupby('YEARS_BINNED').mean()
#
#         ## customer
#         # Age information into a separate dataframe
#         age_data_id = df_customer_id[['DAYS_BIRTH']]
#         age_data_id['DAYS_BIRTH'] = -age_data_id['DAYS_BIRTH']
#         age_data_id['YEARS_BIRTH'] = age_data_id['DAYS_BIRTH'] / 365
#         # Bin the age data
#         age_data_id['YEARS_BINNED'] = pd.cut(age_data_id['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
#         # Group by the bin and calculate averages
#         age_groups_id = age_data_id.groupby('YEARS_BINNED').mean()
#         age_groups_id = age_groups_id.reset_index()
#         df_n_bin = age_groups_id[age_groups_id['DAYS_BIRTH'].notnull()].index
#         n_bin = df_n_bin[0]
#
# ###############
# ###############
# ###############
#         # Subplot
#         #int(math.ceil(len(features)/2))
#         nrows = 2
#         ncols = 2
#
#         # Create subplots, using 'domain' type for pie charts
#         specs = [[{'type':''}, {'type':''}],
#                  [{'type':''}, {'type':''}]]
#
#         fig = make_subplots(rows=nrows, cols=ncols,
#                             #subplot_titles=feature
#                             )
#
#
#         # Plot
#         nrows_ = 0
#         ncols_ = 0
#         i = 0
#
#         for nrows_ in range(nrows) :
#             for ncols_ in range(ncols) :
#                 if i < len(features) :
#
#                     #
#                     feature = features[i]
#
#                     #
#                     #customer_pos_x = df_customer_id[feature].values[0]
#                     #customer_pos_x = round(customer_pos_x, 1)
#
#                     #
#                     x = 100 * age_groups['TARGET']
#
#                     fig.add_trace(go.Histogram(x=x, name=feature),
#                                   nrows_ + 1,
#                                   ncols_ + 1)
#
#                     #
#                     #fig.update_xaxes(title_text=str(feature) + ': ' + str(customer_pos_x), row=nrows_ + 1, col=ncols_ + 1)
#                     #fig.update_yaxes(title_text="Count", row=nrows_ + 1, col=ncols_ + 1)
#                     fig.update_traces(showlegend=False)
#
#                     #
#                     fig.update_layout(autosize=False,
#                                       width=500,
#                                       height=700,
#                                       )
#
#                 i = i + 1
#                 ncols_ = ncols_ + 1
#             nrows_ = nrows_ + 1
#
# ###############
# ###############
#
#         #plot
#         # fig = plt.figure(figsize = (12, 8))
#         # ax = fig.subplots()
#         # # Graph the age bins and the average of the target as a bar plot
#         # bars = ax.bar(age_groups.index.astype(str),
#         #                100 * age_groups['TARGET'],
#         #                edgecolor = 'k')
#         #
#         # bars[n_bin].set_color('r')
#         #
#         #
#         # ax.tick_params(axis='x', rotation=45)
#         # ax.set_xlabel('Age Group (years)')
#         # ax.set_ylabel('Failure to Repay (%)')
#         # ax.set_title('Failure to Repay by Age Group')
#         #
#         #
#         #
#         #
#         #
#         # st.pyplot(fig)

    #else:
##################
##################
    # Subplot
    #int(math.ceil(len(features)/2))
    nrows = 2
    ncols = 2

    # Create subplots, using 'domain' type for pie charts
    # specs = [[{'type':''}, {'type':''}],
    #          [{'type':''}, {'type':''}]]

    fig = make_subplots(rows=nrows, cols=ncols)

    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        for ncols_ in range(ncols) :
            if i < len(features) :
                #
                feature = features[i]
                customer_pos_x = round(df_customer_id[feature].values[0], 1)

                # remove outilers in data
                x_0 = inter_quartile_method_function(df_customers_0[feature])
                x_1 = inter_quartile_method_function(df_customers_1[feature])

                # traces
                fig.add_trace(go.Histogram(x=x_0,
                                           name=str(df_customers_0[hist_bar_group][0])
                                           ),
                              nrows_ + 1,
                              ncols_ + 1)
                fig.add_trace(go.Histogram(x=x_1,
                                           name=str(df_customers_1[hist_bar_group][0])
                                           ),
                              nrows_ + 1,
                              ncols_ + 1)
                fig.add_vline(x=customer_pos_x,
                              line_dash="dot",
                              line_color="black",
                              line_width=1.5,
                              row=nrows_ + 1,
                              col=ncols_ + 1)

                # parameters
                fig.update_xaxes(title_text=str(feature) + ': ' + str(customer_pos_x),
                                 row=nrows_ + 1,
                                 col=ncols_ + 1)
                fig.update_yaxes(title_text="Count",
                                 row=nrows_ + 1,
                                 col=ncols_ + 1)
                fig.update_traces(showlegend=True,
                                  opacity=0.75)
                fig.update_layout(autosize=False,
                                  width=500,
                                  height=700,
                                  barmode='overlay',
                                  legend_tracegroupgap = 150,
                                  legend_title=hist_bar_group)

            i = i + 1
            ncols_ = ncols_ + 1
        nrows_ = nrows_ + 1


    st.plotly_chart(fig, use_container_width=True)

    return



def graph_box_plotly_function(cat_feature, num_feature, df_customers, df_customer_id, box_group):
    """
    Displays dynamic grid of box plots

    Args:
        cat_feature list(str): selected categorical features by end-user
        num_feature list(str): selected numerical features by end-user
        binary_feature (str): selected binary feature by end-user
        df_customers (pd.DataFrame): database
        df_customer_id (pd.DataFrame): customer data
        target (str): selected target ("APPROVED" or "DENIED") by end-user
    Returns
        -
    """

    #
    df_customers[num_feature] = inter_quartile_method_function(df_customers[num_feature])

    # Find the order
    my_order_labels = (df_customers.groupby(by=[cat_feature])[num_feature]
                            .median()
                            .sort_values(ascending=True, kind='quicksort')
                            .iloc[::-1]
                            .index)
    #
    customer_label = df_customer_id[cat_feature].values[0]
    customer_value = df_customer_id[num_feature].values[0]


    fig = px.box(df_customers,
                 x=cat_feature,
                 y=num_feature,
                 color=box_group,
                 #width=1000,
                 height=600)

    fig.add_hline(y=customer_value, line_width=2, line_dash="dot", line_color="black")
    fig.add_vline(x=customer_label, line_width=2, line_dash="dot", line_color="black")

    # parameters
    fig.update_xaxes(categoryorder='array',
                     categoryarray=my_order_labels,
                     title_text= str(cat_feature) + ': ' + str(customer_label))

    fig.update_yaxes(title_text=str(num_feature)  + ': ' + str(customer_value))

    fig.update_layout(showlegend=True)

    #
    st.plotly_chart(fig, use_container_width=True)

    return


def graph_scatter_plotly_function(num_feature_1, num_feature_2, df_customers, df_customer_id, scatter_group):
    """
    Displays dynamic grid of scatter plots

    Args:
        num_feature_1 list(str): selected numercial features 1 by end-user
        num_feature_2 list(str): selected numerical features 2 by end-user
        binary_feature (str): selected binary feature by end-user
        df_customers (pd.DataFrame): database
        df_customer_id (pd.DataFrame): customer data
        target (str): selected target ("APPROVED" or "DENIED") by end-user
    Returns
        -
    """

    #
    df_customers[num_feature_1] = inter_quartile_method_function(df_customers[num_feature_1])
    df_customers[num_feature_2] = inter_quartile_method_function(df_customers[num_feature_2])

    #
    df_customers_0 = df_customers[df_customers[scatter_group] == 0]
    df_customers_1 = df_customers[df_customers[scatter_group] == 1]


    customer_value_1 = df_customer_id[num_feature_1].values[0]
    customer_value_2 = df_customer_id[num_feature_2].values[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_customers_0[num_feature_1],
                             y=df_customers_0[num_feature_2],
                             name='ACCEPTED',
                             mode="markers",
                             ))
    fig.add_trace(go.Scatter(x=df_customers_1[num_feature_1],
                             y=df_customers_1[num_feature_2],
                             name='DENIED',
                             mode="markers",
                             ))


    fig.add_hline(y=customer_value_2, line_width=2, line_dash="dot", line_color="black")
    fig.add_vline(x=customer_value_1, line_width=2, line_dash="dot", line_color="black")

    # parameters
    fig.update_xaxes(title_text= str(num_feature_1) + ': ' + str(customer_value_1))
    fig.update_yaxes(title_text=str(num_feature_2)  + ': ' + str(customer_value_2))

    fig.update_layout(showlegend=True)


    st.plotly_chart(fig, use_container_width=False)

    return




def customer_profil_function(df_customer_id, customer_id, features):
    """
    Displays radar plot of customer profil

    Args:
        df_customers (pd.DataFrame): database
        df_customer_id (pd.DataFrame): customer data
        features (str): features to display
    Returns
        -
    """
    df = df_customer_id[features]
    new_list = [s.replace("_RADAR", "") for s in features]
    df[new_list] = df

    customer_id = df.index.values[0]


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=df.iloc[0].tolist(),
                                  theta=new_list,
                                  fill='toself',
                                  name=customer_id,
                                  thetaunit = "radians",
                                 )
                 )

    fig.update_traces(line_color='lightgreen')
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,
                                                 range=[0, 1],
                                                )),
                                                  showlegend=True,
                                                  width=600,
                                                  height=600,
                                                  title="",
                                                )

    #fig.update_traces(line_color = "magenta")

    st.plotly_chart(fig, use_container_width=True)



    return
