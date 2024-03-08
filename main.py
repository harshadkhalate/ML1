from typing import List
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
with open('./abcd.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

symptomList = ['eye_irritation','running_nose','stuffy_nose','watery_eyes','sneezing','itchy_nose','itchy_throat','inflamed_throat','watery_stools','frequent_bowel_movements','abdomen_pain','nausea','bloating','bloody_stools','fever','headachae','more_intense_pain','fatigue','dry_cough','sore_throat','cough','vomiting','heartburn','indigestion','change_in_apetite','anemia','rashes','pain_behind_eyes','pain_in_joints','feeling_of_discomfort','low energy','cough_with_mucus','greenish_yellow_bloody_mucus','shortness_of_breath','chills','sweating','shallow_breathing','chest_pain']

disease = ['Allergy','Diarrhea','cold_and_flue','Stomachae','Dengue','Malaria','Pneumonia']


class Symptomclass(BaseModel):
    symptoms: List[str]


app = FastAPI()


@app.post("/predict")
def func(s: Symptomclass):
    input = []
    for i in symptomList:
        if (i in s.symptoms):
            input.append(int(1))
        else:
            input.append(int(0))
    input = pd.DataFrame([input], columns=symptomList)
    pred = disease[int(loaded_model.predict(input))]
    return {
        'disease': pred
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)