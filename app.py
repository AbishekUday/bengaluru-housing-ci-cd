# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V6DmOx6CEE7vmUifi5lm8dVM08DgI0_C
"""

from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the pipeline
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(total_sqft: float, bath: int, balcony: int, location: str, area_type: str, size: str):
    input_data = pd.DataFrame([{
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "location": location,
        "area_type": area_type,
        "size": size
    }])

    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}