#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[8]:


news_classifier_3 = load_model("model3/", compile = True)
news_classifier_3.summary()

news_classifier_2 = load_model("model2/", compile = True)
news_classifier_2.summary()

news_classifier_1 = load_model("model1/", compile = True)
news_classifier_1.summary()


# In[9]:


#'There is no muslim regiment in India' #fake 
# Facebook is failing in global disinformation battle says former employee #fake
# Yoko reveals "I had an affair with Hilary # fake
#Joe Biden still holds significant lead #real
#AI could offer big energy savings for office towers #real
#Grant will help researchers prevent apple fire blight US #real

#input = sys.argv[1]
input = 'There is no muslim regiment in India'
test = []


f=open('vocab.json', 'r')
vocab=json.loads(f.read())

lemm = WordNetLemmatizer()
one = re.sub('[^a-zA-Z]', " ", input)
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


#print("model 1 output 1: "+str(y_11)+" 0: "+str(y_10))
#print("model 2 output 1: "+str(y_21)+" 0: "+str(y_20))
#print("model 3 output 1: "+str(y_31)+" 0: "+str(y_30))
#print(y)

y_1 = (y_11+2*(y_21+y_31))/5
y_0 = 1-y_1

#print("1: "+str(y_1)+" 0: "+str(y_0))
if(y_1>y_0):
    print("fake")
else:
    print("real")
    


#if y == 1:
#    print("fake")
#else:
#    print("real")


# In[ ]:




