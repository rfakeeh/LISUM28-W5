# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:50:39 2023

@author: ranaf
"""


from numba import njit
import pickle
import re
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from mlforecast import MLForecast
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error as MAPE

import plotly_express as px

data_path = 'data/'

df = pd.read_csv(data_path+'Cab_Data.csv')
df['Date of Travel'] = pd.to_datetime(df['Date of Travel'], origin='1899-12-30', unit='D')
df['Profit of Trip'] = df['Price Charged'] - df['Cost of Trip']
df = df[['Transaction ID', 'Date of Travel', 'Company', 'City', 'Profit of Trip']]
df = df.round(2)

top_3_cities = df['City'].value_counts()[:3].index.tolist()
print(top_3_cities)

df = df[df['City'].isin(top_3_cities)]
print(df.head(3), end='\n\n')

df1 = df.groupby(['Date of Travel','Company','City'], as_index=False).agg(Total_Profit = ('Profit of Trip','sum'))
print(df1.head(3), end='\n\n')

pvt = pd.pivot_table(data=df1, index=['City','Company'], columns=['Date of Travel'], values=['Total_Profit'])
pvt = pvt.fillna(0)
df2 = pvt.stack().reset_index()
print(df2.head(3), end='\n\n')

df2_cat = df2.select_dtypes(include=['object'])
print(df2_cat.head(3), end='\n\n')

df2_cat = df2_cat.replace(r'[\s]+','_',regex=True)
df2_dummies = pd.get_dummies(df2_cat, drop_first=True)
print(df2_dummies.head(3), end='\n\n')

df3 = df2.join(df2_dummies)
df3 = df3.rename(columns={'Date of Travel':'ds'})
df3['unique_id'] = df3['Company']+'_'+df3['City']
df3['unique_id'] = df3['unique_id'].replace(r'[\s]+','_',regex=True)
df3 = df3.drop(['Company','City'], axis=1)
print(df3.head(3), end='\n\n')

static_features = df2_dummies.columns.tolist()
print(static_features, end='\n\n')

ts_profit = df3.copy()
ts_profit = ts_profit.rename(columns={'Total_Profit':'y'})
print(ts_profit.head(3), end='\n\n')

ts_profit_train = ts_profit[ts_profit['ds']<'2018-10-01']
ts_profit_test = ts_profit[ts_profit['ds']>='2018-10-01']

models = {
    'LinearRegression':make_pipeline(StandardScaler(),MinMaxScaler(),LinearRegression()),
    'Lasso':make_pipeline(StandardScaler(),MinMaxScaler(),Lasso(random_state=0)),
    'Ridge':make_pipeline(StandardScaler(),MinMaxScaler(),Ridge(random_state=0)),
    'KNeighborsRegressor':make_pipeline(StandardScaler(),MinMaxScaler(),KNeighborsRegressor()),
    'SVR':make_pipeline(StandardScaler(),MinMaxScaler(),SVR()),
    'RandomForestRegressor':RandomForestRegressor(random_state=0),
}

fcst = MLForecast(
    models=models,
    freq='D',
    lags=[1,7,30,365],
    date_features=['day','dayofweek','week','month','quarter','year'],
    num_threads=8
)

preprocessed = fcst.preprocess(ts_profit_train)
print(preprocessed.head(3), end='\n\n')

fcst.fit(ts_profit_train, time_col='ds', id_col='unique_id', target_col='y', static_features=static_features)
print(fcst, end='\n\n')

unique_ids = ts_profit_train['unique_id'].unique().tolist()
print(unique_ids, end='\n\n')


with open('profit_model.pkl', 'wb') as file:
    pickle.dump((fcst,unique_ids), file)

profit_model = None
unique_ids = None
with open('profit_model.pkl', 'rb') as file:
    (profit_model,unique_ids) = pickle.load(file)

days = 120
company = 'Yellow Cab'
company = re.sub(r'[\s]+', '_', company)
uids = [uid for uid in unique_ids if company in uid]
print(days, company, uids, unique_ids, end='\n\n')


ts_profit_pred = profit_model.predict(days, ids=unique_ids)
ts_profit_pred = ts_profit_pred[ts_profit_pred['unique_id'].str.contains(company)]

print(ts_profit_pred.info())


ts_profit_pred = ts_profit_pred.rename(columns={'RandomForestRegressor':'profit'})
ts_profit_pred = ts_profit_pred.copy()[['unique_id','ds','profit']]
print(ts_profit_pred.head(10), end='\n\n')


ts_profit_pred['unique_id'] = ts_profit_pred['unique_id'].str.replace(company+'_','')
ts_profit_pred = ts_profit_pred.rename(columns={'unique_id':'city'})
print(ts_profit_pred.head(3), end='\n\n')


ts_profit_pred_summary = ts_profit_pred.groupby(['ds'],as_index=False).agg(Total=('profit','sum'),
                                                                           Mean=('profit','mean'),
                                                                           Min=('profit','min'),
                                                                           Max=('profit','max'))
ts_profit_pred_summary = ts_profit_pred_summary.melt(id_vars='ds')
ts_profit_pred_summary = ts_profit_pred_summary.copy().rename(columns={'value':'profit'})
print(ts_profit_pred_summary.head(3), end='\n\n')

fig1 = px.line(ts_profit_pred_summary, 
               x='ds', 
               y='profit',
               color='variable',  
               height=600,
               title=company+'\'s Total Profit For Next '+str(days)+' Day(s)')

_html = fig1.to_html(full_html=False)
with open('prfit_line.html','w',encoding='utf-8') as file:
    file.write(_html)
    
fig2 = px.line(ts_profit_pred, 
               x='ds', 
               y='profit',
               color='city', 
               height=600,
               title=company+'\'s Total Profit For Next '+str(days)+' Day(s) By City')

_html = fig2.to_html(full_html=False)
with open('prfit_by_city_line.html','w',encoding='utf-8') as file:
    file.write(_html)
    
fig3 = px.pie(ts_profit_pred, 
             names='city', 
             values='profit', 
             height=600,
             title=company+'\'s Total Profit For Next '+str(days)+' Day(s) By City')

_html = fig3.to_html(full_html=False)
with open('prfit_by_city_pie.html','w',encoding='utf-8') as file:
    file.write(_html)