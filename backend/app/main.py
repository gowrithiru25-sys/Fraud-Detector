from fastapi import FastAPI          # import the tool that builds our web server
from pydantic import BaseModel        # import the tool that lets u define what data looks like
import joblib                         # import the tool that loads our saved model file
import pandas as pd                   # import the tool that organizes data into tables

app = FastAPI()                       # create the web server application

model = joblib.load("model.pkl")      # load the trained model from disk into memory

class Transaction(BaseModel):         # define a shape that describes what a valid request looks like
    Time: float                       # must include a field called Time and  must be float
    V1: float                         # must include a field called V1 and it must be a float
    V2: float                         
    V3: float     
    V4: float                    
    V5: float                         
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float        


@app.post("/predict")                 # when someone sends data to the /predict URL
def predict(transaction: Transaction):  # run this function using the transaction shape to check their data
    input_data = pd.DataFrame([transaction.dict()])   # turn their data from the rquest into a one-row table
    prediction = model.predict(input_data)[0]          # ask the model fraud 1 or not 0
    probabilities = model.predict_proba(input_data)[0]  # ask the model how confident are you in each option
    confidence = probabilities[1] if prediction == 1 else probabilities[0]  # grab the confidence for whichever answer won

    return {
    "prediction": "fraud" if prediction == 1 else "legitimate",
     "confidence": round(float(confidence), 4)
    }