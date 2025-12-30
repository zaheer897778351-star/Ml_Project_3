import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocess_path = os.path.join("artifacts","preprocess.pkl")
            model = load_object(file_path=model_path)
            preprocess = load_object(file_path=preprocess_path)
            transformed = preprocess.transform(features)
            result = model.predict(transformed)

            return result


        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,reviewText:str):
        self.reviewText = reviewText

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "reviewText":[self.reviewText]
            }
            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e,sys)
        