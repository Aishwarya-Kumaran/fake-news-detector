from flask import Flask,request,request,jsonify
from flask_cors import CORS

import re
import sys
#from calc imp
import pandas as pd
import json
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)
CORS(app)

news_classifier_3 = load_model("model3/", compile = True)
news_classifier_3.summary()

news_classifier_2 = load_model("model2/", compile = True)
news_classifier_2.summary()

news_classifier_1 = load_model("model1/", compile = True)
news_classifier_1.summary()



@app.route('/')
def home():
    return '<h1> Welcome </h1>'

@app.route('/detectFake', methods = ['POST'])
def detectFakeNews():

    print(request.json['News'])

    input = request.json['News']
    test = []


    f=open('vocab.json', 'r')
    vocab=json.loads(f.read())

    lemm = WordNetLemmatizer()
    one = re.sub('[^a-zA-Z]', " ", str(input))
    one = one.lower()
    one = one.split()
    one = [lemm.lemmatize(word) for word in one if word not in stopwords.words("english")]
    one = " ".join(one)
    print(one)

    encoded_test= []

    one_copy = one.split()
    #print(one_copy)
    for i in one_copy:
        if i in vocab:
            encoded_test.append(vocab[i])
        else:
            encoded_test.append(0)
    test = []
    test.append(encoded_test)

    final_test = pad_sequences(test, maxlen = 356, padding = 'pre')
    y_11 = news_classifier_1.predict_proba(final_test)
    y_10 = 1 - y_11

    y_21 = news_classifier_2.predict_proba(final_test)
    y_20 = 1 - y_21


    y_31 = news_classifier_3.predict_proba(final_test)
    y_30 = 1 - y_31

    y_1 = (y_11+2*(y_21+y_31))/5
    y_0 = 1-y_1

    if(y_1 > y_0):
        return jsonify({
        "isFake" : False
    })
    else:
        return jsonify({
        "isFake" : True
    })

if __name__ == '__main__' :
    app.run(debug=True)