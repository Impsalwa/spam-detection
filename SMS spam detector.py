# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 19:12:26 2021

@author: Salwa
"""
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #for ttf-idf and word count methods
from wordcloud import WordCloud
import numpy as np
import matplotlib as plt
import pandas as pd

#load the data
data = pd.read_csv(r'C:\Users\Salwa\Documents\NLP udemy formation\spam detector\spam.csv', encoding='ISO-8859-1')
print(data.head()) #we notice some empty columns 
#to clean those colomns 
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
print(data.head())

#rename the columns 
data.columns = ['Labels', 'Data']
#print(data.head())

#create a binary labels 
data['binary_labels'] = data['Labels'].map({'ham': 0, 'spam': 1})
#convert it to numpy array
Y = data['binary_labels'].to_numpy()
#print(data.head())

#split the data 
datatrain, datatest, Ytrain, Ytest = train_test_split(data['Data'], Y, test_size=0.33)

#feature extraction TDIDF 
#featurizer = TfidfVectorizer(decode_error='ignore')
#Xtrain = featurizer.fit_transform(datatrain)
#Xtest = featurizer.fit_transform(datatest)
#print(Xtrain)
#print(Xtest)

#feature extraction Word counting 
featurizer = CountVectorizer(decode_error='ignore')
Xtrain = featurizer.fit_transform(datatrain) #both fitting and tranforming 
Xtest = featurizer.transform(datatest) #just the transforming 
print(Xtrain)
print(Xtest)

#create the model 
model = MultinomialNB()
#train the model 
model.fit(Xtrain,Ytrain)
#evaluate the model
score_train = model.score(Xtrain, Ytrain)
score_test = model.score(Xtest, Ytest)
print("train score is : ", score_train)
print("Test score is: ", score_test)