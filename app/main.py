from fastapi import FastAPI, Query
from typing import List
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
import torch
import pytorch
from pytorch import PytorchMultiClass as PytorchMultiClass


app = FastAPI()
ce_target=load('../models/ce_target.joblib')
sc =load('../models/sc.joblib')
le =load('../models/le.joblib')


model = PytorchMultiClass(6)
PATH = "../models/beeroracle_final_normal.pt"
model.load_state_dict(torch.load(PATH))
model.eval()




@app.get("/")
def read_root():
    return  {"""BeerOracle will predict the beer style based on the collated customer feedback from BeerAdvocate.\n
             Expected inputs are: Aroma review: [review_aroma]\n
             Appearance review: [review_appearance]\n
             Palate review: [review_palate] \n
             Taste review: [review_taste] \n
             Alcohol volume: [beer_abv] \n
             Brewery id: [brewery_id] \n
            \n
             Go to /beer/type to see the prediction for a single customer rating\n
             Go to /beers/type to see the prediction for multiple customer ratings\n
            \n
            Prediction model is available on github.com/epo-maker/BeerOracle"""}

            


@app.get('/health', status_code=200)
def healthcheck():
    return 'BeerOracle is ready to predict the type of poison'

def format_features(review_aroma:int, review_appearance: int, review_palate: int, review_taste: int, beer_abv:int, brewery_id:int):
    return {
        'Aroma review': [review_aroma],
        'Appearance review': [review_appearance],
        'Palate review': [review_palate],
        'Taste review': [review_taste],
        'Alcohol volume': [beer_abv],
        'Brewery id': [brewery_id],
         }


@app.get("/beer/type")

def predict(review_aroma:int, review_appearance: int, review_palate: int, review_taste: int, beer_abv:int, brewery_id:int):
    features = format_features(review_aroma, review_appearance, review_palate, review_taste, beer_abv, brewery_id)
    obs = pd.DataFrame(features)
    obs.rename( columns={'Brewery id' :'id_new'}, inplace=True )
    obs['id_new']=ce_target.transform(obs['id_new'])
    obs = torch.Tensor(sc.transform(obs))
    pred = model(obs).argmax(1)
    pred = le.inverse_transform(np.array(pred))
    return JSONResponse(pred.tolist())

def format_features_multiple(review_aroma=[int], review_appearance=[int], review_palate=[int], review_taste=[int], beer_abv=[int], brewery_id=[int]):
    return {
        'Aroma review': np.array(review_aroma).astype(int),
        'Appearance review':np.array(review_appearance).astype(int),
        'Palate review':np.array(review_palate).astype(int),
        'Taste review': np.array(review_taste).astype(int),
        'Alcohol volume': np.array(beer_abv).astype(int),
        'Brewery id': np.array(brewery_id).astype(int),
       }


@app.get("/beers/type")
def predict_multiple(review_aroma:List[int] = Query(...), review_appearance:List[int] = Query(...), review_palate:List[int] = Query(...), review_taste:List[int] = Query(...), beer_abv:List[int] = Query(...), brewery_id:List[int] = Query(...)):
    features = format_features_multiple(review_aroma, review_appearance, review_palate, review_taste, beer_abv, brewery_id)
    obs = pd.DataFrame({'Aroma review':features['Aroma review'], 
                 'Appreance review':features['Appearance review'],
                 'Palate review':features['Palate review'],
                 'Taste review':features['Taste review'],
                 'Alcohol volume':features['Alcohol volume'],
                 'id_new':features['Brewery id']})
    obs['id_new']=ce_target.transform(obs['id_new'])
    obs = torch.Tensor(sc.transform(obs))
    pred = model(obs).argmax(1)
    pred = le.inverse_transform(np.array(pred))
    return JSONResponse(pred.tolist())
    