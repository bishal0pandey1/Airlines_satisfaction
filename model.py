import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


#data preprocessing
dataset=pd.read_csv(r'Data/train.csv')
dataset.isnull().sum()

#features removal
dataset.drop(columns=['Unnamed: 0','id','Gender','Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1,inplace=True)


#encoding
le_Customer=LabelEncoder()
le_Travel=LabelEncoder()
le_Class=LabelEncoder()
le_Satisfaction=LabelEncoder()

dataset['Customer Type']=le_Customer.fit_transform(dataset['Customer Type'])
dataset['Type of Travel']=le_Travel.fit_transform(dataset['Type of Travel'])
dataset['Class']=le_Class.fit_transform(dataset['Class'])
dataset['satisfaction']=le_Satisfaction.fit_transform(dataset['satisfaction'])

#Training data
x_train=dataset.drop('satisfaction',axis=1)
y_train=dataset.satisfaction

#standarization
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)



#importing testing data
dataset1=pd.read_csv(r'Data/test.csv')

#dropping unnecessary columns
dataset1.drop(columns=['Unnamed: 0','id','Gender','Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1,inplace=True)

#encoding testing data
dataset1['Customer Type']=le_Customer.fit_transform(dataset1['Customer Type'])
dataset1['Type of Travel']=le_Travel.fit_transform(dataset1['Type of Travel'])
dataset1['Class']=le_Class.fit_transform(dataset1['Class'])
dataset1['satisfaction']=le_Satisfaction.fit_transform(dataset1['satisfaction'])

#Testing data
x_test=dataset1.drop('satisfaction',axis=1)
y_test=dataset1.satisfaction

#standarization of testing data
x_test=scaler.fit_transform(x_test)



#Training the model
model=RandomForestClassifier()
model.fit(x_train,y_train)

#making predictions
y_pred=model.predict(x_test)

#accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy:',{accuracy})

joblib.dump(model,'output_model/RandomForest.sav')