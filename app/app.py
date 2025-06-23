from fastapi import FastAPI, Request
import torch
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessor import process_data
from app.utils import load_checkpoint
from app.preprocessor import process_data

app=FastAPI()
checkpoint=load_checkpoint(path="app/checkpoint.pt")

@app.post("/predict")
async def predict(request: Request):
    json_data=await request.json()
    df=pd.DataFrame(json_data)
    data=process_data(df=df)
    input_data=torch.Tensor(np.array(data))
    h,w=input_data.shape
    input_data=input_data.reshape((-1,h,w))
    with torch.no_grad():
        prediction=checkpoint(input_data).item()
    return {"prediction":prediction}