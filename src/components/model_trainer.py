import os
import sys
import pandas as pd
from dataclasses import dataclass


from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    results_path = os.path.join("artifacts","results.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            logging.info("Split training and test input data")
            models = {
                'Xgboost' : XGBRegressor(n_jobs = -1),
                'DecisionTrees': DecisionTreeRegressor(),
                'RandomforestRegressor': RandomForestRegressor(n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(n_jobs=-1),
            }

            model_scores = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            results = pd.DataFrame(model_scores)
            results.to_csv(self.model_trainer_config.results_path,index=False)
            logging.info(f"Training and Evaluation Complete!")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models['RandomforestRegressor']
            )

            predicted=models['RandomforestRegressor'].predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
