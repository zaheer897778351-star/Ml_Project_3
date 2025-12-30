import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import sys
import os
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


w= WordNetLemmatizer()

def clean_txt(file):
    try:
        logging.info("Called clean txt")
        if isinstance(file,pd.DataFrame):
            file = file.iloc[:,0]

        file = file.astype(str)
        corpus = []
        for i in file:
            text = i.lower()
            text = re.sub(r'(http|https|ftp|ssh)://[^\s]+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = BeautifulSoup(text, 'lxml').get_text()
            text = re.sub(r'[^a-z0-9 ]', ' ', text)
            words = [w for w in text.split() if w not in set(stopwords.words('english'))]
            text = ' '.join(words)
            text = re.sub(r'\s+', ' ', text).strip()
            corpus.append(text)
        return pd.Series(corpus)
    
    except Exception as e:
        raise CustomException(e,sys)

def lemmatize_text(text):
    try:
        logging.info("called lemmatize txt")
        if isinstance(text, pd.DataFrame):
            text = text.iloc[:, 0]

        text = text.astype(str)
        corpus = []
        for i in text:
            words = i.split()
            words = [w.lemmatize(word) for word in words]
            words = ' '.join(words)
            corpus.append(words)
        return pd.Series(corpus)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path,obj):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train,X_test,y_train,y_test,models,param):
    try:
        logging.info("entered into model training")
        score = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_pred_test = model.predict(X_test)

            rank = accuracy_score(y_test,y_pred_test)
            score[list(models.keys())[i]] = rank

            return score


    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)


    except Exception as e:
        raise CustomException(e,sys)