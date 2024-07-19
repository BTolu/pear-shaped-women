#!/usr/bin/env python
# coding: utf-8


import uvicorn
from fastapi import FastAPI
import pickle 
from pydantic import BaseModel
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


@app.post('/predict')
def predict(data: Measurements):
    data_dict = data.dict()
    full_length = data_dict['full_length']
    shoulder = data_dict['shoulder']
    bust = data_dict['bust']
    waist = data_dict['waist']

    prediction = model.predict([[full_length, shoulder, bust, waist]])
    return {"prediction": prediction}
    
    # return {prediction}

# test_data = Measurements(full_length=1, shoulder=10, bust=10, waist=10)
# prediction = predict(test_data)
# print(prediction)
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.1', port=8000)