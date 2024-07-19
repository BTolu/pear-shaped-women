#!/usr/bin/env python
# coding: utf-8


import uvicorn
from fastapi import FastAPI
import pickle 
from pydantic import BaseModel
import json
import warnings
warnings.filterwarnings('ignore')

class Measurements(BaseModel):
    full_length : float
    shoulder : float
    bust : float
    waist : float

with open("rf_classifier.pkl", 'rb') as f:
    model = pickle.load(f)
    

app = FastAPI()

target_names = ["Half Length", "Bust Length", "Hip Length 1", "Hip Length 2", "Hip", "Knee Length", "Round Knee", "Crotch Line", "Crotch Extension", "Bust Span", "Lap", "Long Sleeve Length", "Long Sleeve Round", "Short Sleeve Length", "Short Sleeve Round", "3/4 Sleeve Length", "3/4 Sleeve Round", "Cap Sleeve Length", "Cap Sleeve Round" ]

@app.post('/predict')
def predict(data: Measurements):
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
    
    # return {prediction}

# test_data = Measurements(full_length=1, shoulder=10, bust=10, waist=10)
# prediction = predict(test_data)
# print(prediction)
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.1', port=8000)