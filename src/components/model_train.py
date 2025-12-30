from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model
from sklearn.metrics import accuracy_score
from src.utils import save_object

@dataclass
class ModeltrainConfig:
    model_train_file_path = os.path.join("artifacts","model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModeltrainConfig()

    def initiate_model_training(self,X_train,y_train,X_test,y_test):
        try:
            X_train = X_train
            X_test = X_test
            y_train = y_train
            y_test = y_test

            models = {
                "logistic":LogisticRegression(),
                "multi" : MultinomialNB()
            }
            param_grids = {
                "logistic": {
                            "C": [0.01, 0.1, 1, 10],
                            "solver": ["liblinear"],
                            "max_iter": [1000],
                        "class_weight": [None, "balanced"]
                         },
                "multi": {
                "alpha": [0.01, 0.1, 0.5, 1.0],
                "fit_prior": [True, False]
                }
            }

            model_reports:dict = evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models,param=param_grids)
            best_model_socre = max(sorted(model_reports.values()))
            best_model_name = list(models.keys())[
                list(model_reports.values()).index(best_model_socre)
            ]
            best_model = models[best_model_name]
            if best_model_socre < 0.6:
                raise CustomException("no best model found ")
            logging.info("best model found")

            predicted = best_model.predict(X_test)
            r_score = accuracy_score(y_test,predicted)
            logging.info("model training completed")

            save_object(file_path=self.model_train_config.model_train_file_path,
                        obj=best_model)

            return(
                r_score,best_model
            )


        except Exception as e:
            raise CustomException(e,sys)