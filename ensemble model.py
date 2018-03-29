# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:32:19 2018

@author: Hongtao Liu
"""

import numpy as np
import pandas as pd

model1=pd.read_csv('lasso_persales.csv',index_col='index')
model2=pd.read_csv('lasso_sales.csv',index_col='index')
model3=pd.read_csv('xgb_persales.csv',index_col='index')
model4=pd.read_csv('xgb_sales.csv',index_col='index')
model5=pd.read_csv('Timeseries.csv',index_col='index')


ensemble_model=model1
ensemble_model.sales=0.2*model1.sales+0.1*model2.sales+0.6*model3.sales+0.05*model4.sales+0.05*model5.sales
ensemble_model.to_csv('ensemble.csv')