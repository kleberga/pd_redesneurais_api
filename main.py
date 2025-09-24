from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from neuralprophet import load
from sklearn.preprocessing import MinMaxScaler
import torch

app = FastAPI()

model = load("model/neuralprophet_model.np")
df = pd.read_json('database/df_hist.json',  orient='records', lines=True)
df = df.sort_values('datetime').copy()
ult_data = pd.to_datetime(df["datetime"].max())
df = df.rename(columns={'datetime': 'ds', 'close': 'y'})

class ForecastInput(BaseModel):
    selic: list
    cambio: list
    prod_petroleo: list
    cotacao_petroleo: list

@app.post("/forecast")
def forecast(data: ForecastInput):

    df_forecast = pd.DataFrame({"ds": [ult_data+pd.Timedelta(days=1)], "selic": data.selic, "cambio": data.cambio, "prod_petroleo": data.prod_petroleo,
                       "cotacao_petroleo": data.cotacao_petroleo, "y": [None]})
    
    df_2 = pd.concat([df, df_forecast], ignore_index=True)

    df_2 = df_2.sort_values('ds').copy()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features = df_2.drop(columns=['y','ds'])
    target = df_2['y']

    # Scale features
    features = pd.DataFrame(scaler_x.fit_transform(features), columns=features.columns)

    # Scale target while preserving missing values
    target_scaled = target.copy()
    non_missing_mask = target.notna()
    target_scaled[non_missing_mask] = scaler_y.fit_transform(
        target[non_missing_mask].values.reshape(-1, 1)
    ).ravel()

    # Combine scaled features and target
    df_3 = pd.concat([df_2['ds'], target_scaled.rename("y"), features], axis=1)

    df_3 = df_3.tail(11)

    forecast = model.predict(df_3)
    yhat = forecast[['ds','yhat1']].iloc[-1]
    df_3['y'].iloc[-1] = yhat['yhat1']

    target_2 = df_3['y']

    target_2_actual = scaler_y.inverse_transform(target_2.values.reshape(-1, 1))
    df_with_forecast = pd.DataFrame([{yhat['ds'], target_2_actual[-1][0]}], columns=['ds','yhat'])

    return df_with_forecast.to_dict(orient="records")

@app.get("/ult_data")
def get_ult_data():
    return ult_data