from flask import Flask,render_template,request
import numpy as np 
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            reviewText = request.form.get('reviewText')
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        res = predict_pipeline.Predict(pred_df)

        if res == 1:
            results = "Good review"
        else:
            results = "Bad reveiew"
        
        return render_template('home.html',results=results)
    
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
