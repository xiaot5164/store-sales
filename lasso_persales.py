# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 08:55:10 2018

@author: Hongtao Liu

target: per sales
"""


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import  Lasso
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')

df['openlag']=df.open.shift(1)
df['openforward']=df.open.shift(-1)
df['promlag']=df.prom.shift(1)
df['promforward']=df.prom.shift(-1)
df['chrislag']=df.chris.shift(1)
df['chrisforward']=df.chris.shift(-1)
df['easterlag']=df.easter.shift(1)
df['easterforward']=df.easter.shift(-1)
df['phlag']=df.ph.shift(1)
df['number']=df.index
df['phforward']=df.ph.shift(-1)
#df['openlag2']=df.open.shift(2)
#df['openforward2']=df.open.shift(-2)
df['year']=pd.to_datetime(df.date).dt.year.astype(int)
df['weekday']=pd.to_datetime(df.date).dt.weekday.astype(int)
df['day']=pd.to_datetime(df.date).dt.day.astype(int)
df = pd.get_dummies(df, columns=['year', 'weekday', 'month','day'])
df.drop('date',axis=1,inplace=True)
df.openforward[941]=1
df.promforward[941]=0
df.chrisforward[941]=0
df.phforward[941]=0
df.easterforward[941]=1

train=df.iloc[:642]
validation=df.iloc[642:792]
test=df.iloc[792:]


train['sales']=pd.to_numeric(train['sales'])
train.loc[(train['sales']<0.1),'sales']=0
train.loc[(train['customers']<0.1),'customers']=0
train['persales']=train.sales/train.customers

validation['sales']=pd.to_numeric(validation['sales'])
validation.loc[(validation['sales']<0.1),'sales']=0
validation.loc[(validation['customers']<0.1),'customers']=0
validation['persales']=validation.sales/validation.customers

test['sales']=pd.to_numeric(train['sales'])
test.drop('sales',axis=1,inplace=True)


Xtr=train.loc[(train['sales']!=0),]
Xtr.drop(['sales','persales'],axis=1,inplace=True)
ytr=train.loc[(train['sales']!=0),'persales']

Xva=validation.loc[(validation['sales']!=0),]
Xva.drop(['sales','persales'],axis=1,inplace=True)
yva=validation.loc[(validation['sales']!=0),'persales']


dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xva, label=yva)

lasso_model = Lasso(alpha=0.004)
lasso_model.fit(Xtr,ytr)
yva_predict=lasso_model.predict(Xva)
RMSE=(np.dot((yva_predict*Xva.customers-yva*Xva.customers),(yva_predict*Xva.customers-yva*Xva.customers).T)/125)**0.5
print(RMSE)




coef=lasso_model.coef_
y_te=lasso_model.predict(test) 
y_te=pd.DataFrame(y_te,columns=['sales'])
y_te.set_index(test.index,inplace=True)
y_test=pd.concat([y_te,test['customers']],axis=1,ignore_index = False)
y_test.loc[(y_test['customers']<1),'sales']=0
#try to move the prediction because the mean of residuals is not 0
y_test['sales']=y_test['sales']*y_test['customers']+102
y_test.drop('customers',axis=1,inplace=True)

y_test.to_csv('lasso_persales.csv')

residual=yva*Xva.customers-102-yva_predict*Xva.customers
plt.hist(residual,20)
residual.mean()
residual.std()  