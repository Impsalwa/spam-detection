# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:55:21 2021

@author: Salwa
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np 

#load the data
data = pd.read_csv('spambase.data').values #transform it to a matrix 
#print(data)

#shiffle the data to be deffrent every time split the train/ test data
np.random.shuffle(data)
#print(data)

#split the data to train and test 

X= data[:, :48] #all rows and 48 columns
Y = data[:, -1] #all rows and from 49 to last columns

#xtrain and ytrain and xtest ytest split

Xtrain = X[:-100,] #first 100 rows
Ytrain = Y[:-100,]
Xtest = X[-100:,] #last 100 rows
Ytest = Y[-100:,]

#initiate the model 
model = MultinomialNB()
model_svm = SVC()
model_boostclf = AdaBoostClassifier()
#train the model
model.fit(Xtrain, Ytrain)
model_svm.fit(Xtrain, Ytrain)
model_boostclf.fit(Xtrain, Ytrain)
#prediction of the models 
prediction_nb= model.predict(Xtest)
prediction_svc = model_svm.predict(Xtest)
prediction_adaboost = model_boostclf.predict(Xtest)

print("Prediction for Naive bayes : ", prediction_nb)
print("Prediction for svc :", prediction_svc)
print("Prediction for AdaBoost :", prediction_adaboost)

#calcutae the score and evaluate the models 
score = model.score(Xtest, Ytest)
score_svm = model_svm.score(Xtest, Ytest)
score_boostclf = model_boostclf.score(Xtest, Ytest)

print("Classification rate for NB: ", score)
print("Classification rate for SVC: ", score_svm)
print ("Classification rate for AdaboostClasifier: ",score_boostclf)



