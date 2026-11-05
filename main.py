
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal


# 1. Load Machine Learning Model
model = joblib.load(
    r"C:\Users\MAJID MAJEED\OneDrive\Desktop\mentalheALTHPREDICTOR\Mental_Health_Model.pkl"
)


# 2. Create FastAPI Application
app = FastAPI()


# 3. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. Define Input Data Validation
class StudentData(BaseModel):

    Age: int = Field(..., ge=10, le=100)

    Gender: Literal['Male', 'Female']

    Country: str

    Academic_Level: Literal[
        'Undergraduate',
        'Graduate',
        'High School'
    ]

    Most_Used_Platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
        'Twitter', 'YouTube', 'TikTok', 'LINE',
        'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ]

    Purpose_Of_Use: Literal[
        'Networking',
        'Education',
        'Entertainment',
        'News'
    ]

    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24)

    Daily_Unlocks: int = Field(..., ge=0)

    Study_Hours: float = Field(..., ge=0, le=24)

    Physical_Activity_Hours: float = Field(..., ge=0, le=24)

    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)

    Stress_Level: Literal[
        'Medium',
        'Low',
        'Very High',
        'High'
    ]


# 5. Define Prediction Response
class PredictionResponse(BaseModel):

    predicted_mental_health_score: float


# 6. Define Top Countries
top_countries = [
    'Other',
    'India',
    'USA',
    'Canada',
    'Australia',
    'UK',
    'Germany',
    'Mexico',
    'Turkey',
    'France'
]


# 7. Home Route
@app.get("/")
def greet():

    return {
        "message": "Mental Health Predictor API is running!"
    }


# 8. Prediction Route
@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):

    # Group Countries
    country_group = (
        data.Country
        if data.Country in top_countries
        else 'Other'
    )

    # Create Input DataFrame
    input_row = pd.DataFrame([{

        'Age': data.Age,

        'Gender': data.Gender,

        'Country': data.Country,

        'Academic_Level': data.Academic_Level,

        'Most_Used_Platform': data.Most_Used_Platform,

        'Purpose_Of_Use': data.Purpose_Of_Use,

        'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,

        'Daily_Unlocks': data.Daily_Unlocks,

        'Study_Hours': data.Study_Hours,

        'Physical_Activity_Hours': data.Physical_Activity_Hours,

        'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,

        'Stress_Level': data.Stress_Level,

        'Grouped_Country': country_group

    }])


    # Generate Prediction
    prediction = model.predict(input_row)[0]


    # Return Prediction
    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )
