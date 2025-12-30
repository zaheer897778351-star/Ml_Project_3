from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
from src.utils import clean_txt,lemmatize_text,save_object
import pandas as pd
import numpy as np


@dataclass
class DataTransformConfig:
    data_transform_file_path = os.path.join("artifacts","preprocess.pkl")

class DataTransform:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()

    def data_transform(self):
        try:
            logging.info("Now entered in data transform")
            cate = ['reviewText']

            cate_pipeline = Pipeline(
                steps=[
                    ("clean",FunctionTransformer(clean_txt,validate=False)),
                    ("lemmatize",FunctionTransformer(lemmatize_text,validate=False)),
                    ("wordtovec",TfidfVectorizer(
                        max_features=2500,
                        ngram_range=(1,2)
                    ))
                ]
            )

            preprocess = ColumnTransformer([
                ("final",cate_pipeline,cate)
            ])

            return preprocess

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train,test):
        try:
            train_data = pd.read_csv(train)
            test_data = pd.read_csv(test)
            target = 'rating'

            preprocess_obj = self.data_transform()

            input_train_data = train_data.drop(columns=[target],axis=1)
            input_train_target_data = train_data[target]

            input_test_data = test_data.drop(columns=[target],axis=1)
            input_test_target_data = test_data[target]

            input_train_transformed = preprocess_obj.fit_transform(input_train_data)
            input_test_transformed = preprocess_obj.transform(input_test_data)

            # train_df = np.c_[input_train_transformed,np.array(input_train_target_data)]
            # test_df = np.c_[input_test_transformed,np.array(input_test_target_data)]

            save_object(file_path=self.data_transform_config.data_transform_file_path,
                        obj=preprocess_obj)

            return (
            input_train_transformed,
            input_train_target_data,
            input_test_transformed,
            input_test_target_data
        )


        except Exception as e:
            raise CustomException(e,sys)