import os
import re
import math
import warnings
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import folium
import datetime
import nltk
import gensim
import pyLDAvis 
import imblearn
import random
import shap
import time
import plotly.express as px
from lime import lime_tabular
from IPython import display
from pyLDAvis import sklearn
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from datetime import datetime
from collections import defaultdict, Counter
from folium import IFrame
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, anneal, Trials, STATUS_OK, space_eval
from matplotlib.ticker import MaxNLocator
from folium.plugins import HeatMap
from sklearn.metrics import (accuracy_score, 
                             plot_confusion_matrix, 
                             confusion_matrix, 
                             classification_report, 
                             f1_score, 
                             roc_auc_score, 
                             recall_score, 
                             precision_score, 
                             roc_curve, 
                             fbeta_score, 
                             make_scorer)
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, AllKNN, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline as pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.impute import SimpleImputer
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (train_test_split, 
                                     RepeatedStratifiedKFold, 
                                     StratifiedKFold, 
                                     cross_validate, 
                                     cross_val_score,
                                     learning_curve)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from PIL import Image, ImageOps, ImageFilter
from scipy.sparse import coo_matrix, csr_matrix 
from yellowbrick.cluster import KElbowVisualizer
from geopy.geocoders import Nominatim
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

#init_notebook_mode(connected = True)

def info(dataframe):
    """Prints dataframe parameters
    
    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        Prints parameters: number of columns, number of rows, rate of missing values
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values : " + str(dataframe.isnull().mean().mean()*100) + " %")
    print("")
    
    
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
    
    
    
class ClassifierBestParamsSearch:
            
        
    def __init__(self, dataframe,
                       numerical_transformers, 
                       categorical_transformers, 
                       target,
                       classifier_type,
                       max_evals,
                       score,
                       **kwargs):

        """
        General function for classifier model creation, optimization and evaluation
        Args:
            dataframe (pd.Dataframe): input
            numerical_transformers: steps for numerical features treatments
            categorical_transformers: steps for categorical features treatments
            classifier_type (sklearn function): name of classifier algorithm
            max_evals (int): number of hyperopt maximum evaluations
        Returns:
            -
        """ 
        self.dataframe = dataframe
        self.X = dataframe.drop(columns=[target])
        self.y = dataframe[[target]]
        self.dataframe_fit = dataframe.drop(columns=[target])
        self.numerical_transformers = numerical_transformers
        self.categorical_transformers = categorical_transformers
        self.target = target
        self.classifier_type = classifier_type
        self.max_evals = max_evals
        self.score = score
        self.classifier_type_string = str(classifier_type)
        
    
    def define_general_pipeline(self, params):
        """
        Builds data transformation, resampling and classification pipeline
        
        Args:
            params (dict)
        Returns
            model (scklearn pipeline)
        """
    
        # Preprocessor pipeline
        # Create sub-pipelines for numartical & categorical features
        numerical_preprocessor = make_pipeline(*self.numerical_transformers)
        categorical_preprocessor = make_pipeline(*self.categorical_transformers)
        # Associate both pipelines
        preprocessor = ColumnTransformer(
            transformers=[("numerical", 
                           numerical_preprocessor, 
                           make_column_selector(dtype_include="number")), 
                          ("categorical", 
                           categorical_preprocessor, 
                           make_column_selector(dtype_include=["object"]))
                         ]
        )
        

        # oversampling
        smt = SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=42)
        # undersampling
        rus = RandomUnderSampler(sampling_strategy=1)
                               
        # General pipeline
        model = PipelineImb(steps=[("pre-processor", preprocessor), 
                                   ("oversampling", smt),
                                   ("undersampling", rus),
                                   ("classifier", self.classifier_type(**params))
                                  ]
                           )
   
        return model


    def space_definition(self):
        """
        Define hyperopt search spaces
        
        Args:
            -
        Returns
            space (dict)
        """
        if "DummyClassifier" in self.classifier_type_string:
            
            print("dummy space")
            
            # Define search space
            space = {
                     "strategy": hp.choice("strategy", ['most_frequent','prior','stratified','uniform','constant']),
                     #'random_state': None,
                     #'constant': None
                    }  

        elif "LogisticRegression" in self.classifier_type_string:
            
            # Define search space
            space = {'penalty': hp.choice('penalty', ['l2']), #****
                     'tol': hp.loguniform('tol', 1e-8, 1e-1),
                     'C': hp.uniform('C', low=0.01, high=100),
                     'fit_intercept': hp.choice('fit_intercept', [True, False]),
                     'class_weight': hp.choice('class_weight', [None, 'balanced']),
                     #'random_state': 1234,
                     'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']) 
                     #'max_iter': [90,100,110],
                     #'multi_class':'auto', #{‘auto’, ‘ovr’, ‘multinomial’}
                     #'warm_start': False,
                     #'n_jobs': None,
                     #'l1_ratio':None,  
                    }  
              
        elif "XGBRFClassifier" in self.classifier_type_string:
            
            # Define search space
            space = {'colsample_bynode': hp.choice('colsample_bynode', [0.4, 0.5, 0.6, 0.7, 0.8, 1]),
                      'learning_rate': hp.uniform('learning_rate', low=0, high=2),
                      'max_depth': hp.choice('max_depth', [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                      'num_parallel_tree': hp.choice('num_parallel_tree', [50, 70, 90, 110, 130, 150]),
                      'objective': hp.choice('objective', ['reg:squarederror',
                                                           'reg:squaredlogerror',
                                                           'reg:logistic']),               
                      'subsample': hp.choice('subsample', [0.1, 0.3, 0.5, 0.7, 0.9, 1]),
                      'tree_method': hp.choice('tree_method', ['auto',
                                                               'exact',
                                                               'approx',
                                                               'hist',
                                                               'gpu_hist']),
                    }
            
            ''''reg:pseudohubererror',
                                                            'binary:logistic',
                                                            'binary:logitraw',
                                                            'binary:hinge',
                                                            'count:poisson',
                                                            'max_delta_step',
                                                            'survival:cox',
                                                            'survival:aft',
                                                            'aft_loss_distribution',
                                                            'multi:softmax',
                                                            'multi:softprob',
                                                            'rank:pairwise',
                                                            'rank:ndcg',
                                                            'rank:map',
                                                            'reg:gamma',
                                                            'reg:tweedie']),'''
            
        elif "XGBClassifier" in self.classifier_type_string:
            
            # Define search space
            space = {'learning_rate': hp.uniform('learning_rate', 0, 0.5),
                     'max_depth': hp.choice('max_depth', [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                     'gamma': hp.uniform ('gamma', 0, 10),
                     'reg_alpha' : hp.quniform('reg_alpha', 40, 180, 1),
                     'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
                     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                     'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
                     'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 250, 300]),
                     'seed': 0,
                    }
                     
        return space
    

    # Definition of the objective function
    def tuning_objective(self, args):
        """
        Define objective function
        
        Args:
            -
            args (dict):
        Returns
            loss function: -fbeta_score
        """
        
        #
        n_features = len(self.X.columns)
                
        # split data beteween train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            train_size=0.80, 
                                                            test_size=0.2, 
                                                            stratify=self.y, 
                                                            random_state=1234)

        #print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        
        # define model
        model = self.define_general_pipeline(args)
        # train model
        model.fit(X_train, y_train)
        # predict test set
        y_pred = model.predict(X_test)
            
        #
        fbeta_score_ = fbeta_score(y_test, y_pred, beta=10)

        print(f"fbeta_score: {fbeta_score_}")
        print(args)
        
        return {"loss": -fbeta_score_, "status": STATUS_OK}
        
    
    def trials_classifier(self):
        """
        Dispalys trials params 
        
        Args:
            -
            args (dict):
        Returns
            best_params: dict
        """
        
        # Initialize trials object
        trials = Trials()
        search_space = self.space_definition()
        print(search_space)
        print(self.max_evals)
                
        #    
        best = fmin(fn=self.tuning_objective,
                    space = search_space, 
                    algo=tpe.suggest, 
                    max_evals=self.max_evals, 
                    trials=trials,
                    rstate=np.random.default_rng(123)
                   )

        #
        best_params = space_eval(search_space, best)
        print(best_params)
        
        return best_params
    
    
    def executes_algo_function(self, params):
        """
        Dispalys trials params 
        
        Args:
            -
            params (dict): model parameters
        Returns
            stats (list): scklearn metrics
            model (scklearn pipeline): fit and trained model
            X_test (pd.DataFrame):
            y_test (pd.DataFrame):
            y_pred (np.array): X_test predicted labels
        """
        #
        n_features = len(self.X.columns)
                
        # split data beteween train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            train_size=0.80, 
                                                            test_size=0.2, 
                                                            stratify=self.y, 
                                                            random_state=1234)
        
        
        # define model
        model = self.define_general_pipeline(params)
        
        # train model
        model.fit(X_train, y_train)
        # predict test set
        y_pred = model.predict(X_test)
                           
        # imblearn
        # compute repartition of y labels on train set
        counter = Counter(y_train.squeeze())
        # preprocessing and resampling data
        X_train_resample, y_train_resample = model[:-1].fit_resample(X_train, y_train)
        # compute repartition of y labels after resampling on train set
        counter_2 = Counter(y_train_resample.squeeze())
        
        print("* Imblearn:")
        print('y_train labels before resampling: ' + str(counter))
        print('y_train labels after resampling: ' + str(counter_2) + '\n')
        
        # Stats
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        fbeta = fbeta_score(y_test, y_pred, beta=10)
        
        print('* Stats:')
        print('roc_auc_score: ' + str(roc_auc))
        print('f1_score: ' + str(f1))
        print('accuracy_score: ' + str(accuracy))
        print('recall_score: ' + str(recall))
        print('precision_score: ' + str(precision))
        print('fbeta_score: ' + str(fbeta))
        print()
        
        # metrics
        stats = [roc_auc, 
                 f1, 
                 accuracy, 
                 recall, 
                 precision, 
                 fbeta]
        
        # feature importance
        self.feature_importances_function(self.X.columns.tolist(), 
                                          model, 
                                          n_features)
        

        return stats, model, self.X, self.y, X_test, y_test, y_pred 
            
        
    def executes_algo_function_lime(self, params):
        """
        Displays LIME plot
        
        Args:
            params (dict): model parameters
        Returns
            -
        """
        
        #
        n_features = len(self.X.columns)
        
        # define model
        model = self.define_general_pipeline(params)
        
        #
        X = model.named_steps['pre-processor'].fit_transform(self.X)
        X = X[:,:n_features]
        y = self.y.values
              
        
        # split data beteween train and test
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            train_size=0.80, 
                                                            test_size=0.2, 
                                                            stratify=y, 
                                                            random_state=12)

        #print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                
        #
        X_train_imb, y_train_imb = model.named_steps['oversampling'].fit_resample(X_train, y_train)
        X_train_imb, y_train_imb = model.named_steps['undersampling'].fit_resample(X_train_imb, y_train_imb)

        # define classifier
        #clf = self.define_classifier_pipeline(params)
        
        # train classifier
        model.named_steps['classifier'].fit(X_train_imb, y_train_imb)
        
        # display stats
        print("* Sets stats:")
        print("Test  Accuracy: %.2f"%model.named_steps['classifier'].score(X_test, y_test))
        print("Train Accuracy: %.2f"%model.named_steps['classifier'].score(X_train, y_train))
        print("")
        print("* Confusion Matrix:")
        print(confusion_matrix(y_test, model.named_steps['classifier'].predict(X_test)))
        print()
        print("* Classification Report:")
        print(classification_report(y_test, model.named_steps['classifier'].predict(X_test)))
        
        
        # LIME
        print("\n* Lime:")
        explainer = lime_tabular.LimeTabularExplainer(X_train_imb, 
                                                      mode="classification",
                                                      class_names=np.unique(y_train),
                                                      feature_names=self.X.columns.tolist())
        #
        preds = model.named_steps['classifier'].predict(X_test)


        # right predictions    
        idx = random.randint(0,len(preds))
        print('rows:' + str(idx))
        print("Prediction: ", model.named_steps['classifier'].predict(X_test[idx].reshape(1,-1))[0])
        print("Actual: ", y_test[idx][0])

        explanation = explainer.explain_instance(X_test[idx], 
                                                 model.named_steps['classifier'].predict_proba, 
                                                 num_features=n_features)
        explanation.show_in_notebook()
            
        if "Dummy" in self.classifier_type_string:
            return
        elif "LogisticRegression" in self.classifier_type_string:
            importance = model.named_steps['classifier'].coef_[0]
        elif "XGB" in self.classifier_type_string:
            importance = model.named_steps['classifier'].feature_importances_
        
        print("\n\n* Weights")
        with plt.style.context("ggplot"):
            fig = plt.figure(figsize=(12,16))
            plt.barh(range(len(importance)), 
                     importance, 
                     color=["red" if coef<0 else "green" for coef in importance])
            plt.gca().invert_yaxis()
            plt.yticks(range(len(importance)), 
                       self.X.columns.tolist())
            plt.title("Weights")
            plt.show()

        print("Explanation Local Prediction: ","1" if explanation.local_pred<0.5 else "0")
        print("Explanation Global Prediction Probability: ", explanation.predict_proba)
        print("Explanation Global Prediction: ", y_test[np.argmax(explanation.predict_proba)][0])

        return
    
    def executes_algo_function_shap(self,
                                    params,
                                    n_features,
                                    n_features_y_test,
                                    plot_type_string,
                                    feature,
                                    label):
        """
        Displays SHAP plot
        
        Args:
            params (dict): model parameters,
            n_features (int): number of features to consider,
            n_features_y_test (int): ,
            plot_type_string (str): type of SHAP plot,
            feature: target feature,
            label:
        Returns
            -
        """
        
        # define model
        model = self.define_general_pipeline(params)
                
        # data
        X = model.named_steps['pre-processor'].fit_transform(self.X)
        X = X[:,:n_features]
        y = self.y.values
      
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            train_size=0.85,
                                                            test_size=0.15,
                                                            stratify=y,
                                                            random_state=123,
                                                            shuffle=True)

        #print("Train/Test Sizes : ",X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        #
        X_train_imb, y_train_imb = model.named_steps['oversampling'].fit_resample(X_train, y_train)
        X_train_imb, y_train_imb = model.named_steps['undersampling'].fit_resample(X_train_imb, y_train_imb)

        # train classifier
        model.named_steps['classifier'].fit(X_train_imb, y_train_imb)

        print("* Sets stats:")
        print("Test  Accuracy : ", model.named_steps['classifier'].score(X_test, y_test))
        print("Train Accuracy : ", model.named_steps['classifier'].score(X_train_imb, y_train_imb))
        
        #
        print("\n* Shap:")
        sub_sampled_train_data = shap.sample(X_train_imb, 600, random_state=0)
        subsampled_test_data = X_test[0:n_features_y_test,:]

        start_time = time.time()
        explainer = shap.KernelExplainer(model.named_steps['classifier'].predict_proba, 
                                         sub_sampled_train_data)
        shap_values = explainer.shap_values(subsampled_test_data, l1_reg="aic")
        elapsed_time = time.time() - start_time

        # explain first sample from test data
        print("Kernel Explainer SHAP run time:", round(elapsed_time, 3) , " seconds. ")
        print("Calssifier:", self.classifier_type_string)
        print("SHAP expected value:", explainer.expected_value)
        print("Model mean value:", model.named_steps['classifier'].predict_proba(X_train).mean(axis=0))
        #print("Model prediction for test data", model.named_steps['classifier'].predict_proba(subsampled_test_data))
        shap.initjs()
        pred_ind = 0
        
        if plot_type_string == "force" :
            shap.force_plot(explainer.expected_value[1], 
                            shap_values[1][label],
                            subsampled_test_data[0],
                            feature_names=self.X.columns[:n_features],
                            text_rotation=45,
                            matplotlib=True,
                            show=False)
        
        elif plot_type_string == "summary" :
            shap.summary_plot(shap_values[label],
                              subsampled_test_data,
                              feature_names=self.X.columns[:n_features],
                              max_display=10)
            
        elif plot_type_string == "embedding" :
            shap.embedding_plot(feature,
                                explainer.shap_values(subsampled_test_data)[label],
                                feature_names=self.X.columns[:n_features])
        
        elif plot_type_string == "dependence" :
            shap.dependence_plot(feature,
                                 explainer.shap_values(subsampled_test_data)[label],
                                 features=subsampled_test_data,
                                 feature_names=self.X.columns[:n_features])
            
        elif plot_type_string == "decision" :    
            shap.multioutput_decision_plot(explainer.expected_value.tolist(),
                                           explainer.shap_values(subsampled_test_data),
                                           row_index=0,
                                           feature_names=self.X.columns[:n_features].tolist(),
                                           highlight = [1])
                                           
        elif plot_type_string == "bar" :    
            shap.bar_plot(explainer.shap_values(subsampled_test_data[0])[label],
                          feature_names=self.X.columns[:n_features],
                          max_display=len(self.X.columns[:n_features]))

            
        elif plot_type_string == "waterfall" :    
            shap.waterfall_plot(explainer.shap_values,
                                max_display=len(self.X.columns[:n_features]),
                                show=True)
                               
            
        '''clf_explainer = shap.LinearExplainer(model.named_steps['classifier'], X_train_imb)
        
        print(clf_explainer)
        
        sample_idx = 0
        shap_vals = clf_explainer.shap_values(X_test[sample_idx].reshape(-1, 1))
        
        print(clf_explainer.expected_value)

        #val1 = clf_explainer.expected_value + shap_vals.sum()
        #val2 = clf_explainer.expected_value[1] + shap_vals[1].sum()'''

        '''print("Base Value: ", clf_explainer.expected_value)
        print()
        print("Shap Values for Sample %d: "%sample_idx, shap_vals)
        print("\n")
        print("Prediction From Model: ", y_test[model.predict(X_test[sample_idx].reshape(1, -1))[0]])
        #print("Prediction From Adding SHAP Values to Base Value: ", y_test.values[np.argmax([val1, val2, val3])])
        
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        
        shap_values = explainer.shap_values(X_test, l1_reg="aic")
        
        # explain first sample from test data
        print("Kernel Explainer SHAP run time", round(elapsed_time,3) , " seconds. ", current_model["name"])
        print("SHAP expected value", explainer.expected_value)
        print("Model mean value", model.predict_proba(X_train).mean(axis=0))
        print("Model prediction for test data", model.predict_proba(X_test))
        
        shap.initjs()
        pred_ind = 0
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], subsampled_test_data[0], feature_names=self.X.columns.tolist())'''

        return
    

    def feature_importances_function(self, labels, model, n_features):
        """
        Displays feature importances plot
        
        Args:
            params (dict): model parameters,
            n_features (int): number of features to consider,
            n_features_y_test (int): ,
            plot_type_string (str): type of SHAP plot,
            feature: target feature,
            label:
        Returns
            -
        """ 
        
        #
        if "Dummy" in self.classifier_type_string:
            return
        elif "LogisticRegression" in self.classifier_type_string:
            importance = model.named_steps['classifier'].coef_[0]
        elif "XGB" in self.classifier_type_string:
            importance = model.named_steps['classifier'].feature_importances_

        # 
        importance = importance[:n_features]

        # associate labels and importance score
        my_dict = {labels[i]: importance[i] for i in range(len(labels))} 

        # sort dict by importance score
        my_list = sorted(my_dict.items(), key=lambda x:x[1], reverse=True)
        my_dict = dict(my_list)

        # plot feature importance
        print("* Feature Importance:")
        fig, ax = plt.subplots(figsize=(14, 8))
        plt.bar([x for x in range(len(my_dict))], my_dict.values())
        plt.title("Feature Importance")
        plt.ylabel("Contribution in %")
        plt.xlabel("Features")
        ticks = list(range(0, n_features))
        plt.xticks(ticks, my_dict.keys(), rotation=90)
        plt.show()  
        
        # summarize feature importance
        for item in my_dict.items():
            print('Feature: ' + str(item[0]) + ', Score: ' + str(item[1]))
        print("")

        return

   
        
def stats_comparaison_function(metric, order, **kwargs):
    """
    Builds dataframe ranking models by metric perfromance

    Args:
        metric (str): ranking order according to given metric, 
        order (str): ascending or descending,
    Returns
        dataframe (pd.Dataframe) : ranking
    """                         
    # data
    d = {}
    
    for kwarg in kwargs:
        newline = {kwarg: kwargs[kwarg]}
        d.update(newline)

    index = ['roc_auc_score', 
             'f1_score', 
             'accuracy_score', 
             'recall_score', 
             'precision_score',
             'fbeta_score',
            ]
             

    # dataframe
    df = pd.DataFrame(data=d, index=index).transpose()

    order = order.lower()
    
    #
    if order == 'ascending':
        order = True 
    elif order == 'descending':
        order = False 
    
    #
    return df.sort_values(by=metric, axis=0, ascending=order, inplace=False, kind='quicksort')



def plot_confusion_matrix_function(y_true, y_pred, title):
    """ Plot confusion matrix
    Args:
        y_true list(str):
        y_pred list(int):
        title (str): 
    Returns:
        -
    """
    # confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # plot the heatmap for correlation matrix
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(cf_matrix.T, 
                 square=True, 
                 annot=True, 
                 annot_kws={"size": 17},
                 fmt='.2f',
                 cmap='Blues',
                 cbar=True,
                 ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right", fontsize=17)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=17)
    ax.set_ylabel("predicted labels", fontsize=17)
    ax.set_xlabel("true labels", fontsize=17)

    plt.show()
    
    return
    
    
    
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        X_test, 
                        y_test, 
                        y_pred,
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes[0][0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0][0].set_xlabel("Training examples")
    axes[0][0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator,
                                                                          X,
                                                                          y,
                                                                          cv=cv,
                                                                          n_jobs=n_jobs,
                                                                          train_sizes=train_sizes,
                                                                          return_times=True
                                                                         )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0][0].grid()
    axes[0][0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0][0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0][0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0][0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0][0].legend(loc="best")
    #axes[0][0].set_xticks(fontsize=16)
    #axes[0][0].set_yticks(fontsize=16)
    #axes[0][0].title(r"", fontsize=16)

    # Plot n_samples vs fit_times
    axes[0][1].grid()
    axes[0][1].plot(train_sizes, fit_times_mean, "o-")
    axes[0][1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[0][1].set_xlabel("Training examples")
    axes[0][1].set_ylabel("fit_times")
    axes[0][1].set_title("Scalability of the model")
    #axes[0][1].set_xticks(fontsize=16)
    #axes[0][1].set_yticks(fontsize=16)
    #axes[0][1].title(r"", fontsize=16)

    # Plot fit_time vs score
    axes[1][0].grid()
    axes[1][0].plot(fit_times_mean, test_scores_mean, "o-")
    axes[1][0].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[1][0].set_xlabel("fit_times")
    axes[1][0].set_ylabel("Score")
    axes[1][0].set_title("Performance of the model")
    #axes[1][0].set_xticks(fontsize=16)
    #axes[1][0].set_yticks(fontsize=16)
    #axes[1][0].title(r"", fontsize=16)
    
    
    # plot confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu", cbar=False, ax=axes[1][1])
    axes[1][1].set_xlabel("Predicted Target")
    axes[1][1].set_ylabel("True Target")
    axes[1][1].set_title("Confusion matrix")
    #axes[1][1].set_xticks(fontsize=16)
    #axes[1][1].set_yticks(fontsize=16)
    #axes[1][1].title(r"", fontsize=16)
    

    plt.show()

    return 