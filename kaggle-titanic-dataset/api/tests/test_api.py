from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from app.model import Model
from typing import List
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class dummyModel(Model):
    
    def train(self, train:UploadFile = File(...)):
        print("model trained")
        pass
        
    def predict(self, test):
        pass

class Prediction(BaseModel):
    predictions:List[dict] = []
    
@app.post("/predict", response_model = Prediction)
def predict(test:UploadFile = File(...)):
    print(file)
    train_df = pd.read_csv(test)
    return {"predictions" : [{"Survived" : train_df.Survived[0]}]}

client = TestClient(app=app)
response = client.post(
    "/predict", 
    files={"file": ("test.csv", open("../../data/test.csv", "rb"), "text/csv")}
    )

print(response)
assert response.status_code == 200