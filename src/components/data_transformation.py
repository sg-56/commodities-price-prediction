import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.impute import KNNImputer
from feature_engine.creation import CyclicalFeatures
from feature_engine.encoding import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            #Encoding Cyclic Features i.e (Day,Month,dayofweek,dayofmonth,dayofyear,dayofmonth)
            cyclic_features = ['dayofweek','quarter','month','dayofyear','dayofmonth','weekofyear']
            cat_feature = ['market','commodity']
            price_imputer = Pipeline(
                steps=[
                ("imputer",KNNImputer(n_neighbors=15))
                ]
            )
            cyclic_pipeline=Pipeline(

                steps=[
                ("Cyclic",CyclicalFeatures(variables=cyclic_features,drop_original=True))
                ]
            )
            category_pipeline = Pipeline(
                steps=[("one_hot_encoder",OneHotEncoder(variables=cat_feature))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",price_imputer,['price']),
                ("Cyclic_Pipeline",cyclic_pipeline,cyclic_features),
                ("cat_pipelines",category_pipeline,cat_feature)
                ]

            )
            logging.info(f"Categorical columns: {cat_feature}")
            logging.info(f"Numerical columns: {cyclic_features}")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
