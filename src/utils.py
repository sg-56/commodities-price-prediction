import streamlit as st
import pandas as pd
import os
import sys
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging



def hide_sidebar():
    return st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
        display: none
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

def GetCommodities():
    return pd.read_csv('notebooks/Dataset.csv',usecols=['commodity']).commodity.unique()

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,model,param):
    try:
        gs = GridSearchCV(model,param,cv=5,n_jobs=-1)
        gs.fit(X_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

         #model.fit(X_train, y_train)  # Train model

        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        logging.info(f"Train r2 : {train_model_score}")

        test_model_score = r2_score(y_test, y_test_pred)
        logging.info(f"Test r2 : {test_model_score}")
        return test_model_score

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)