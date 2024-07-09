import os
import sys
from dataclasses import dataclass

#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            logging.info("Split training and test input data")
            model = XGBRegressor()
            params= {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }

            model_score = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             model=model,param=params)
            logging.info(f"Training and Evaluation Complete!")
            logging.info(msg=str(model_score))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            predicted=model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)