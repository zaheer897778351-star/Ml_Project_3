import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
from src.components.data_transform import DataTransform,DataTransformConfig
from src.components.model_train import ModelTrain,ModeltrainConfig


@dataclass
class DataIngestionConfig:
    train_data_ingestion_path:str = os.path.join("artifacts","train.csv")
    test_data_ingestion_path:str = os.path.join("artifacts","test.csv")
    raw_data_ingestion_path:str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_date_ingestion(self):
        try:
            logging.info("Entered into a data initialization")
            df = pd.read_csv('notebook/Data/review_rating.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_ingestion_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_ingestion_path,index=False,header=True)
            logging.info("raw data file created in arifacts raw ")
            train_set,test_set = train_test_split(df,test_size=0.33,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_ingestion_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_ingestion_path,index=False,header=True)
            logging.info("train test data also created")

            return (
                self.data_ingestion_config.train_data_ingestion_path,
                self.data_ingestion_config.test_data_ingestion_path
            )

        except Exception as e:
            raise CustomException(e,sys)    

if __name__=="__main__":
    obj = DataIngestion()
    train_set,test_set = obj.initiate_date_ingestion()
    data_trans = DataTransform()
    X_train, y_train, X_test, y_test= data_trans.initiate_data_transform(train_set,test_set)
    model = ModelTrain()
    print(model.initiate_model_training(X_train,y_train,X_test,y_test))
