from flask import Flask, request, jsonify
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from feature_engineering import hand_features

model = joblib.load("model_mnb.pkl")

application = Flask(__name__)

@application.route('/predict', methods= ['POST'])
def predict():
    if model:
        query= request.json
        query= pd.DataFrame(query)
        query_body= query['body'].copy()
        query_head= query['head'].copy()
        feat= hand_features(query_head, query_body)
        
        prediciton = list(model.predict(feat))
        return jsonify({'prediciton': str(prediciton)})

    else:
        return({"error message": "trained model not found"})
    

if __name__=='__main__':
   
    application.debug= True
    application.run()