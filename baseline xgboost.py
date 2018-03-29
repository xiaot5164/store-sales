# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:49:42 2018

@author: Hongtao Liu

This is the baseline model without feature engineering 

xgboost model
This is Time series question

Target variable is daily sales 
independent varaible: number of customers, date, open or not, promotion or not 
school day or not, chris day or not, easter day or not, public holiday or not 
"""



import pandas as pd
import xgboost as xgb
import numpy as np

df=pd.read_csv('train.csv')
df.head()


#year
df['year']=pd.to_datetime(df.date).dt.year.astype(int)


# 641 training, 150 validation, last 150 test. 
train=df.iloc[:642]
validation=df.iloc[642:792]
test=df.iloc[792:]

# data cleaning, and create weekday
train['sales']=pd.to_numeric(train['sales'])
train.loc[(train['sales']<0.1),'sales']=0
train.loc[(train['customers']<0.1),'customers']=0
train['weekday']=pd.to_datetime(train.date).dt.weekday.astype(int)

validation['sales']=pd.to_numeric(validation['sales'])
validation.loc[(validation['sales']<0.1),'sales']=0
validation.loc[(validation['customers']<0.1),'customers']=0
validation['weekday']=pd.to_datetime(validation.date).dt.weekday.astype(int)

test['sales']=pd.to_numeric(train['sales'])
test['weekday']=pd.to_datetime(train.date).dt.weekday.astype(int)


feature_names=['customers',
               'open',
               'prom',
               'sch',
               'month',
               'chris',
               'easter',
               'ph',
               'weekday',
            ]

# remove the closed day in which the sales is 0
Xtr=train.loc[(train['sales']!=0),feature_names]
ytr=train.loc[(train['sales']!=0),'sales']

Xva=validation.loc[(validation['sales']!=0),feature_names]
yva=validation.loc[(validation['sales']!=0),'sales']


dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xva, label=yva)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


xgb_pars = {'min_child_weight': 6, 'eta': 0.2, 'colsample_bytree': 0.8, 'max_depth':2,
            'subsample': 0.6, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 6000, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=10)

yva_prediction=model.predict(dvalid)