import sys
import pandas as pd
import os
from datetime import datetime
from src.exception import CustomException
from src.utils import load_object,GetCommodities,create_features
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            logging.info("Model and Preprocessor Loaded")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info("Transforming Input Data")
            print(features.shape)
            data_scaled=preprocessor.transform(features)
            print(data_scaled.shape)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        date: datetime,
        market: str,
        commodity:str):

        self.date = date

        self.commodity = commodity

        self.market = market

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "date": [pd.to_datetime(self.date)],
                "commodity": [self.commodity],
                "market": [self.market]
            }
            df = pd.DataFrame(custom_data_input_dict).set_index('date')
            #print(df)
            df = create_features(df)
            return df

        except Exception as e:
            raise CustomException(e, sys)

