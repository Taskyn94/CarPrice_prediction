#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set() 
#Importing Dataset
train=pd.read_csv('train_car.csv',sep=';')
test=pd.read_csv('test_car.csv',sep=';')

train['ENGINE_VOLUME']=train['ENGINE_VOLUME']/10
test['ENGINE_VOLUME']=test['ENGINE_VOLUME']/10
train.dtypes

#Defining dependencies and correlations between Year and price
train.plot(x='YEAR', y='ESTIM_COST', kind='scatter')
train['ESTIM_COST'].corr(train['YEAR']) #-0.445

#train['ENGINE_VOLUME']=train.ENGINE_VOLUME.astype(float)
#Defining dependencies and correlations between Engine Volume and price
train.plot(x='ENGINE_VOLUME', y='ESTIM_COST', kind='scatter')
train['ESTIM_COST'].corr(train['ENGINE_VOLUME']) #0.48


#Preprocessing of FUEL_TYPE
train['FUEL_TYPE'].value_counts()
train['FUEL_TYPE'].loc[train['FUEL_TYPE']=='Petrol']=1
train['FUEL_TYPE'].loc[train['FUEL_TYPE']=='Petrol-CNG']=0.8
train['FUEL_TYPE'].loc[train['FUEL_TYPE']=='Diesel']=0.6
train['FUEL_TYPE'].loc[train['FUEL_TYPE']=='Hybrid']=0.4
train['FUEL_TYPE'].loc[train['FUEL_TYPE']=='CNG']=0.2
train['FUEL_TYPE']=train.FUEL_TYPE.astype(float)

test['FUEL_TYPE'].loc[test['FUEL_TYPE']=='Petrol']=1
test['FUEL_TYPE'].loc[test['FUEL_TYPE']=='Petrol-CNG']=0.8
test['FUEL_TYPE'].loc[test['FUEL_TYPE']=='Diesel']=0.6
test['FUEL_TYPE'].loc[test['FUEL_TYPE']=='Hybrid']=0.4
test['FUEL_TYPE'].loc[test['FUEL_TYPE']=='CNG']=0.2
test['FUEL_TYPE']=test.FUEL_TYPE.astype(float)

#Defining dependencies and correlations between Fueltype and price
train.plot(x='FUEL_TYPE', y='ESTIM_COST', kind='scatter')
train['FUEL_TYPE'].corr(train['ESTIM_COST']) #-0.07



#Prepfrocessing of BODY_TYPE
train['BODY_TYPE'].value_counts()
train.dtypes
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Sedan']=1.4
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Crossover']=1.2
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Hetchbek']=1
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Truck']=0.8
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Universal']=0.6
train['BODY_TYPE'].loc[train['BODY_TYPE']=='MiniVen']=0.4
train['BODY_TYPE'].loc[train['BODY_TYPE']=='Pickup']=0.2
train['BODY_TYPE']=train.BODY_TYPE.astype(float)
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Sedan']=1.4
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Crossover']=1.2
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Hetchbek']=1
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Truck']=0.8
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Universal']=0.6
test['BODY_TYPE'].loc[test['BODY_TYPE']=='MiniVen']=0.4
test['BODY_TYPE'].loc[test['BODY_TYPE']=='Pickup']=0.2
test['BODY_TYPE']=test.BODY_TYPE.astype(float)

#Defining dependencies and correlations between Year and price
train.plot(x='BODY_TYPE', y='ESTIM_COST', kind='scatter')
train['BODY_TYPE'].corr(train['ESTIM_COST']) #corr=-0.18



#Prepfrocessing of TYPE_OF_DRIVE
train['TYPE_OF_DRIVE'].value_counts()
train.dtypes
train['TYPE_OF_DRIVE'].loc[train['TYPE_OF_DRIVE']=='Four-wheel drive']=1.0
train['TYPE_OF_DRIVE'].loc[train['TYPE_OF_DRIVE']=='Front-wheel drive']=0.6
train['TYPE_OF_DRIVE'].loc[train['TYPE_OF_DRIVE']=='Rear drive']=0.2
train['TYPE_OF_DRIVE']=train.TYPE_OF_DRIVE.astype(float)

test['TYPE_OF_DRIVE'].loc[test['TYPE_OF_DRIVE']=='Four-wheel drive']=1.0
test['TYPE_OF_DRIVE'].loc[test['TYPE_OF_DRIVE']=='Front-wheel drive']=0.6
test['TYPE_OF_DRIVE'].loc[test['TYPE_OF_DRIVE']=='Rear drive']=0.2
test['TYPE_OF_DRIVE']=test.TYPE_OF_DRIVE.astype(float)

#Defining dependencies and correlations between Type_of_deive and price
train.plot(x='TYPE_OF_DRIVE', y='ESTIM_COST', kind='scatter')
train['TYPE_OF_DRIVE'].corr(train['ESTIM_COST']) #CORR= 0.32



#Prepfrocessing of 
train['INTERIOR_TYPE'].value_counts()
train.dtypes
train['INTERIOR_TYPE'].loc[train['INTERIOR_TYPE']=='Jewlery']=1.0
train['INTERIOR_TYPE'].loc[train['INTERIOR_TYPE']=='Combined']=0.6
train['INTERIOR_TYPE'].loc[train['INTERIOR_TYPE']=='Leather']=0.2
train['INTERIOR_TYPE']=train.INTERIOR_TYPE.astype(float)

test['INTERIOR_TYPE'].loc[test['INTERIOR_TYPE']=='Jewlery']=1.0
test['INTERIOR_TYPE'].loc[test['INTERIOR_TYPE']=='Combined']=0.6
test['INTERIOR_TYPE'].loc[test['INTERIOR_TYPE']=='Leather']=0.2
test['INTERIOR_TYPE']=test.INTERIOR_TYPE.astype(float)

train.plot(x='INTERIOR_TYPE', y='ESTIM_COST', kind='scatter')
train['INTERIOR_TYPE'].corr(train['ESTIM_COST']) #corr=-0.31



#Preprocessing of Interior Type
train['TRANSM_TYPE'].loc[train['TRANSM_TYPE']=='Automatic']=1
train['TRANSM_TYPE'].loc[train['TRANSM_TYPE']=='Mechanic']=0
train['TRANSM_TYPE']=train.TRANSM_TYPE.astype('int64')

test['TRANSM_TYPE'].loc[test['TRANSM_TYPE']=='Automatic']=1
test['TRANSM_TYPE'].loc[test['TRANSM_TYPE']=='Mechanic']=0
test['TRANSM_TYPE']=test.TRANSM_TYPE.astype('int64')

train.plot(x='TRANSM_TYPE', y='ESTIM_COST', kind='scatter')
train['TRANSM_TYPE'].corr(train['ESTIM_COST']) #corr = 0.22



#Preprocessing COndition
test['AUTO_CONDITION'].value_counts()
train['AUTO_CONDITION'].loc[train['AUTO_CONDITION']=='Excellent']=1.0
train['AUTO_CONDITION'].loc[train['AUTO_CONDITION']=='Good']=0.6
train['AUTO_CONDITION'].loc[train['AUTO_CONDITION']=='Satisfactory']=0.2
train['AUTO_CONDITION']=train.AUTO_CONDITION.astype(float)

test['AUTO_CONDITION'].loc[test['AUTO_CONDITION']=='Exellent']=1.0
test['AUTO_CONDITION'].loc[test['AUTO_CONDITION']=='Good']=0.6
test['AUTO_CONDITION'].loc[test['AUTO_CONDITION']=='Satisfactory']=0.2
test['AUTO_CONDITION']=test.AUTO_CONDITION.astype(float)

train.plot(x='AUTO_CONDITION', y='ESTIM_COST', kind='scatter')
train['AUTO_CONDITION'].corr(train['ESTIM_COST']) #Corr=0.30


#We are droping some features whose corr<|0.3| according to thumb rule
# Corr[Fuel_Type]=0.07 corr[Body_type]=0.18, corr[Trans_type]=0.22
train.drop(['FUEL_TYPE','BODY_TYPE','TRANSM_TYPE'],axis=1,inplace=True)
test.drop(['FUEL_TYPE','BODY_TYPE','TRANSM_TYPE'],axis=1,inplace=True)

test.isnull().values.any()#Checking for NA's
train.to_csv('Cleaned_traincar.csv')
test.to_csv('Cleaned_testcar.csv') 











