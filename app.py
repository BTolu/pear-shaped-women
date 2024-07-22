#!/usr/bin/env python
# coding: utf-8


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle 
from pydantic import BaseModel
import json
import warnings
warnings.filterwarnings('ignore')

# Measurements class
class Measurements(BaseModel):
    full_length : float
    shoulder : float
    bust : float
    waist : float

# load the model
with open("rf_classifier.pkl", 'rb') as f:
    model = pickle.load(f)
    
# Instantiate the app object
app = FastAPI()

# Add CORS middleware
app.add_middleware(CORSMiddleware,
                   allow_origins = ['*'], # allows all origins
                   allow_methods = ['*'], # allows all methods
                   allow_headers = ['*']) # allows all headers 

target_names = ["Half Length", "Bust Length", "Hip Length 1", "Hip Length 2", "Hip", "Knee Length", "Round Knee", "Crotch Line", "Crotch Extension", "Bust Span", "Lap", "Long Sleeve Length", "Long Sleeve Round", "Short Sleeve Length", "Short Sleeve Round", "3/4 Sleeve Length", "3/4 Sleeve Round", "Cap Sleeve Length", "Cap Sleeve Round" ]




@app.post('/predict')
async def predict(data: Measurements):
    data_dict = data.dict()
    full_length = data_dict['full_length']
    shoulder = data_dict['shoulder']
    bust = data_dict['bust']
    waist = data_dict['waist']

    prediction = model.predict([[full_length, shoulder, bust, waist]])
    pred = prediction[0].tolist()
    pred_dict = {}
    for i in range(len(pred)):
        pred_dict[target_names[i]] = pred[i]
    
    return json.dumps(pred_dict)

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.1', port=8000)