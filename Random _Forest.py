import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score#
from sklearn.ensemble import RandomForestClassifier


duration = 15  #after how many days we will predict price
window = 60 #previous day closing price range to be considered.
outputs_30 = []
outputs_60 = []
outputs_entire_test_data = []
dec_max_acc_outputs_30 = []
dec_max_acc_outputs_60 = []
dec_max_acc_outputs_entire = []

data = pd.read_csv("C:\\Users\\HP-PC\\Desktop\\Umass Lowell\\Sem 1\\Machine Learning\\TCS.csv",dtype={'Close': np.float32})

for i in range(len(data) - duration):
   if data.loc[i+duration,'Close'] > data.loc[i,'Close']:
                data.loc[i,'Next Close'] = 1
   else:
                data.loc[i,'Next Close'] = 0
            
for i in reversed(range(window,len(data))):
    for j in range(window+1):
        data.loc[i, 'Close '+str(j)] = data.loc[i-j, 'Close']
previous_closes = []
for j in range(window+1):
    previous_closes.append('Close '+str(j))
   
data_columns = previous_closes.copy()
data_columns.append('Next Close')    
data = data.loc[:,data_columns].dropna()
    
X = data.loc[:,previous_closes]
y = data.loc[:,['Next Close']]
split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]        


for n in range(100,1000,200):
       # print("Duration: "+ str(duration) +" Window: "+ str(window) +" Number of Trees: "+ str(n))
        rf=RandomForestClassifier(n_estimators=n)
        rf.fit(X_train,y_train)
        max_acc = 0
        y_pred = rf.predict(X_test[:int(len(X_test))])    
        acc = accuracy_score(y_test[:int(len(X_test))], y_pred)
        if acc > max_acc:
            days = int(len(X_test)*i / 20)
            max_acc = acc
        outputs_entire_test_data.append({"Duration":duration,"Window":window,"days":days, "Accuracy":max_acc,"Trees":n})
        
for n in range(100,1000,200):
       # print("Duration: "+ str(duration) +" Window: "+ str(window) +" Number of Trees: "+ str(n))
        rf=RandomForestClassifier(n_estimators=n)
        rf.fit(X_train,y_train)
        max_acc = 0
        y_pred = rf.predict(X_test[:61])    
        acc = accuracy_score(y_test[:61], y_pred)
        if acc > max_acc:
            days = 60
            max_acc = acc
        outputs_60.append({"Duration":duration,"Window":window,"days":days, "Accuracy":max_acc,"Trees":n})
        
for n in range(100,1000,200):
       # print("Duration: "+ str(duration) +" Window: "+ str(window) +" Number of Trees: "+ str(n))
        rf=RandomForestClassifier(n_estimators=n)
        rf.fit(X_train,y_train)
        max_acc = 0
        y_pred = rf.predict(X_test[:31])    
        acc = accuracy_score(y_test[:31], y_pred)
        if acc > max_acc:
            days = 30
            max_acc = acc
        outputs_30.append({"Duration":duration,"Window":window,"days":days, "Accuracy":max_acc,"Trees":n})

dec_max_acc_outputs_60.append(sorted(outputs_60, key = lambda i: i['Accuracy'],reverse=True))
dec_max_acc_outputs_entire.append(sorted(outputs_entire_test_data, key = lambda i: i['Accuracy'],reverse=True))        
dec_max_acc_outputs_30.append(sorted(outputs_30, key = lambda i: i['Accuracy'],reverse=True))
