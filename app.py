# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:03:55 2023

@author: ranaf
"""

from flask import Flask, request, jsonify, render_template
import pickle
import re
import json
import plotly_express as px

app = Flask(__name__)

import sys
print(sys.executable)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    
    profit_model = None
    unique_ids = None
    with open('model/profit_model.pkl', 'rb') as file:
        (profit_model,unique_ids) = pickle.load(file)
    
    
    days = int(request.form['days'])
    company = request.form['company']
    
    company = re.sub(r'[\s]+', '_', company)
        
    ts_profit_pred = profit_model.predict(days, ids=unique_ids)
    ts_profit_pred = ts_profit_pred[ts_profit_pred['unique_id'].str.contains(company)]
    
    ts_profit_pred = ts_profit_pred.rename(columns={'RandomForestRegressor':'profit'})
    ts_profit_pred = ts_profit_pred.copy()[['unique_id','ds','profit']]
    ts_profit_pred['unique_id'] = ts_profit_pred['unique_id'].str.replace(company+'_','')
    ts_profit_pred = ts_profit_pred.rename(columns={'unique_id':'city'})
    
    
    ts_profit_pred_summary = ts_profit_pred.groupby(['ds'],as_index=False).agg(Total=('profit','sum'),
                                                                               Mean=('profit','mean'),
                                                                               Min=('profit','min'),
                                                                               Max=('profit','max'))
    ts_profit_pred_summary = ts_profit_pred_summary.melt(id_vars='ds')
    ts_profit_pred_summary = ts_profit_pred_summary.copy().rename(columns={'value':'profit'})
    
    fig1 = px.line(ts_profit_pred_summary,
                   x='ds',
                   y='profit',
                   color='variable',
                   height=600,
                   title=company+'\'s Total Profit For Next '+str(days)+' Day(s)')
                   
    fig2 = px.line(ts_profit_pred, 
                   x='ds', 
                   y='profit',
                   color='city', 
                   height=600,
                   title=company+'\'s Total Profit For Next '+str(days)+' Day(s) By City')
    
    fig3 = px.pie(ts_profit_pred, 
             names='city', 
             values='profit', 
             height=600,
             title=company+'\'s Total Profit For Next '+str(days)+' Day(s) By City')

    return jsonify({'profit_line_plot':json.loads(fig1.to_json()),
                    'profit_per_city_line_plot':json.loads(fig2.to_json()), 
                    'profit_per_city_pie_plot':json.loads(fig3.to_json())})


if __name__ == "__main__":
    app.run(debug=True)