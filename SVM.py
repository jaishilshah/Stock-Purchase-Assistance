# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics

data = pd.read_csv('RELIANCE.csv')
window = 60
duration = 15

for i in range(len(data) - duration):
    if data.loc[i+duration,'Close'] > data.loc[i,'Close']:
        data.loc[i,'Next Close'] = '1'
    else:
        data.loc[i,'Next Close'] = '0'

for i in reversed(range(window,len(data))):
    for j in range(window+1):
        data.loc[i, 'Close '+str(j)] = data.loc[i-j, 'Close']
        
        
data

previous_closes = []
for j in range(window+1):
    previous_closes.append('Close '+str(j))
    
data_columns = previous_closes.copy()
data_columns.append('Next Close')    
data = data.loc[:,data_columns].dropna()

data

X = data.loc[:,previous_closes]
y = data.loc[:,['Next Close']]
split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

svr = svm.SVC(kernel='rbf', C=0.5, gamma=5)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
y_pred

from numpy import argmax
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test[:30], y_pred[:30])
print('30 days:')
print('Accuracy: %.3f' % acc)

acc = accuracy_score(y_test[:60], y_pred[:60])
print('60 days:')
print('Accuracy: %.3f' % acc)

acc = accuracy_score(y_test, y_pred)
print(str(len(y_test)) + ' days:')
print('Accuracy: %.3f' % acc)