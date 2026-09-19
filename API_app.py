from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("demand_forecasting_model.pkl")

class InputData(BaseModel):
    DayOfWeek: int
    Customers: int
    SchoolHoliday: int
    Year: int
    Month: int
    IsPromo: int
    IsPromo2: int
    CompetitionDurationMonths: float

@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame({
            "DayOfWeek": [data.DayOfWeek],
            "Customers": [np.log1p(data.Customers)],  
            "SchoolHoliday": [data.SchoolHoliday],
            "Year": [data.Year],
            "Month": [data.Month],
            "IsPromo": [data.IsPromo],
            "IsPromo2": [data.IsPromo2],
            "CompetitionDurationMonths_log": [np.log1p(data.CompetitionDurationMonths)]
        })

        prediction = model.predict(input_df)

        prediction_value = float(np.expm1(prediction[0]))  

        return {"Predicted_Sales": prediction_value}

    except Exception as e:
        return {"error": str(e)}
