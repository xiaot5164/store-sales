# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:38:34 2018

@author: Hongtao Liu
target: sales
xgboost, dummy variable
"""


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import Ridge, Lasso

df=pd.read_csv('train.csv')

df['openlag']=df.open.shift(1)
#df['openforward']=df.open.shift(-1)
df['promlag']=df.prom.shift(1)
#df['promforward']=df.prom.shift(-1)
df['chrislag']=df.chris.shift(1)
#df['chrisforward']=df.chris.shift(-1)
df['easterlag']=df.easter.shift(1)
#df['easterforward']=df.easter.shift(-1)
df['phlag']=df.ph.shift(1)
df['number']=df.index
#df['phforward']=df.ph.shift(-1)
#df['openlag2']=df.open.shift(2)
#df['openforward2']=df.open.shift(-2)
df['year']=pd.to_datetime(df.date).dt.year.astype(int)
df['weekday']=pd.to_datetime(df.date).dt.weekday.astype(int)

df = pd.get_dummies(df, columns=['year', 'weekday', 'month'])
df.drop('date',axis=1,inplace=True)

train=df.iloc[:642]
validation=df.iloc[642:792]
test=df.iloc[792:]


train['sales']=pd.to_numeric(train['sales'])
train.loc[(train['sales']<0.1),'sales']=0
train.loc[(train['customers']<0.1),'customers']=0


validation['sales']=pd.to_numeric(validation['sales'])
validation.loc[(validation['sales']<0.1),'sales']=0
validation.loc[(validation['customers']<0.1),'customers']=0


test['sales']=pd.to_numeric(train['sales'])
test.drop('sales',axis=1,inplace=True)


Xtr=train.loc[(train['sales']!=0),]
Xtr.drop('sales',axis=1,inplace=True)
ytr=train.loc[(train['sales']!=0),'sales']

Xva=validation.loc[(validation['sales']!=0),]
Xva.drop('sales',axis=1,inplace=True)
yva=validation.loc[(validation['sales']!=0),'sales']


dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xva, label=yva)
dtest=xgb.DMatrix(test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


xgb_pars = {'min_child_weight': 6, 'eta': 0.23, 'colsample_bytree': 0.8, 'max_depth':2,
            'subsample': 0.6, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}


model = xgb.train(xgb_pars, dtrain, 6000, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=10)

y_te=model.predict(dtest) 
y_te=pd.DataFrame(y_te,columns=['sales'])
y_te.set_index(test.index,inplace=True)
y_test=pd.concat([y_te,test['customers']],axis=1,ignore_index = False)
y_test.loc[(y_test['customers']<1),'sales']=0
y_test.drop('customers',axis=1,inplace=True)
y_test.to_csv('xgb_sales.csv')      

yva_predict=model.predict(dvalid)
residual=yva-yva_predict
residual.mean()
residual.std() 