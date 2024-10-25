from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

with open('budget_model.pkl', 'rb') as f:
    budget_model = pickle.load(f)

with open('credits_model.pkl', 'rb') as f:
    credits_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    carbon_emissions: float
    carbon_credits_price: float

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame({
            'Carbon Emissions (kg CO2)': [input_data.carbon_emissions],
            'Carbon Credits Price (INR)': [input_data.carbon_credits_price]
        })
        
        input_scaled = scaler.transform(input_df)

        budget_prediction = budget_model.predict(input_scaled)
        credits_prediction = credits_model.predict(input_scaled)

        return {
            "Carbon Credits Required": credits_prediction[0],  # Carbon credits required
            "predicted Budget Needed (INR)": budget_prediction[0]  # Budget needed
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app with: uvicorn app:app --reload
