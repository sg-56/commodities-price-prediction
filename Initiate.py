from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()
data_transformation=DataTransformation()
X_train,y_train,X_test,y_test,_=data_transformation.initiate_data_transformation(train_data,test_data)
modeltrainer=ModelTrainer()
print(modeltrainer.initiate_model_trainer(X_train=X_train,y_train=y_train,X_test=X_test,y_test = y_test))