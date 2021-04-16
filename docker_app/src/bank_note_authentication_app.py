"""
Date : 15 Apr 2021, 13:29:00

@author : Abhishek R (abhishekr.ar53@gmail.com)
"""

from flask import Flask, request
import pandas as pd 
import numpy as np 
import pickle 
import flasgger
from flasgger import Swagger

#read saved model and load classifier
saved_model =  open("models/random_forest_classifer_model.pkl", "rb")
classifier = pickle.load(saved_model)
    
#Build flass apis
app = Flask(__name__)
Swagger(app)

@app.route("/")
def welcome():
    return "Welcome to the bank note authentication application"

@app.route("/predict", methods=["GET"])
def prediction_service():
    """ Function performs prediction for bank note authentication
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    """
    #read request parameters
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    
    #perform prediciton
    predicted_class = classifier.predict([[variance, skewness, curtosis, entropy]])

    bank_note_result = "Authenticated and Valid Note" if predicted_class == 1  else "Not Authenticated and Invalid Note"

    output_string = f"For Input values {[[variance, skewness, curtosis, entropy]]}<br>Result is:<br>{bank_note_result}"

    return output_string

@app.route("/predict_file", methods=["POST"])
def file_prediction_service():
    """ Function performs prediction for bank note authentication
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
    """
    #read file content request parameters
    df = pd.read_csv(request.files.get("file"))
    #perform prediciton
    predicted_classes = classifier.predict(df)
    output_string = ""
    for (i, value) in enumerate(list(predicted_classes)):
        bank_note_result = "Authenticated and Valid Note" if value == 1  else "Not Authenticated and Invalid Note"
        output_string += f"For Input values {df.loc[i]}<br>Result is:<br>{bank_note_result}<br><br>"
    return output_string


if __name__ == "__main__":
    app.run()
