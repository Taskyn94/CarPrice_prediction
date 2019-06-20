import pandas as pd
import numpy as np

train=pd.read_csv('Cleaned_traincar.csv')
test=pd.read_csv('Cleaned_testcar.csv')
test.drop(['ID'],axis=1,inplace=True)

X=train.iloc[:,1:-1].values
y=train.iloc[:,6].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Making Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

exp=pd.read_csv('ypred.csv',sep=';')
exp['y_pred'].corr(exp['y_test']) # Corr=0.828 acc=82.8%

#let's predict price for our real test set
predPrice=regressor.predict(test)
predPrice.to_csv('testCar_Estimated price.csv')

















