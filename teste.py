import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from neuralprophet import load, NeuralProphet, save
import yfinance as yf
from datetime import date, datetime
import requests

df = pd.read_json('database/df_hist.json',  orient='records', lines=True)
df = df.sort_values('datetime').copy()
ult_data = df["datetime"].max()
hoje = date.today()
delta = hoje - ult_data.date()

if(delta.days > 1):

    petr4 = yf.Ticker('PETR4.SA')

    dados_diarios = petr4.history(start=ult_data+pd.Timedelta(days=1), end=hoje)

    dados_diarios_2 = pd.DataFrame(dados_diarios)
    dados_diarios_2['datetime'] = dados_diarios_2.index
    dados_diarios_2['datetime'] = dados_diarios_2['datetime'].apply(lambda dt: dt.date())
    dados_diarios_2 = dados_diarios_2.reset_index()
    dados_diarios_2.drop(columns=['Date'], inplace=True)

    dados_diarios_2.rename(columns={'Close': 'close'}, inplace=True)

    dados_diarios_2 = dados_diarios_2[['datetime','close']]

    data_ini = pd.to_datetime(min(dados_diarios_2['datetime']))

    data_ini = data_ini.strftime("%d/%m/%Y")

    # baixar os dados de Selic, cambio R$/USD e preço do petróleo
    url_selic = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json&dataInicial='+data_ini
    url_cambio = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=json&dataInicial='+data_ini
    url_preco_petroleo = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='EIA366_PBRENT366')"
    url_prod_petroleo = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1391/dados?formato=json'

    # baixar os dados de selic
    response = requests.get(url_selic)
    if response.status_code == 200:
        data = response.json()
        selic = pd.DataFrame(data)
        selic['data'] = pd.to_datetime(selic['data'], format='%d/%m/%Y')
        selic['selic'] = selic['valor'].astype(float)
        selic.drop(columns=['valor'], inplace=True)
    else:
        print(f"Error: {response.status_code}")

    # baixar os dados de cambio de 2025
    response = requests.get(url_cambio)
    if response.status_code == 200:
        data = response.json()
        cambio = pd.DataFrame(data)
        cambio['data'] = pd.to_datetime(cambio['data'], format='%d/%m/%Y')
        cambio['cambio'] = cambio['valor'].astype(float)
        cambio.drop(columns=['valor'], inplace=True)
    else:
        print(f"Error: {response.status_code}")

    # baixar os dados de produção de petróleo
    response = requests.get(url_prod_petroleo)
    if response.status_code == 200:
        data = response.json()
        prod_petroleo = pd.DataFrame(data)
        prod_petroleo['data'] = pd.to_datetime(prod_petroleo['data'], format='%d/%m/%Y')
        prod_petroleo['prod_petroleo'] = prod_petroleo['valor'].astype(float)
        prod_petroleo.drop(columns=['valor'], inplace=True)
    else:
        print(f"Error: {response.status_code}")

    # fazer interpolação polinomial de ordem para tornar a série diária
    prod_petr_mensal = pd.Series(
        prod_petroleo['prod_petroleo'].astype(float).values,
        index=pd.to_datetime(prod_petroleo['data'].values)
    )

    # transformar a série para periodicidade diária e incluir NaNs nos dias que não tem valor
    prod_petr_diaria = prod_petr_mensal.resample('D').asfreq()

    # interpolar para preencher os dias com NaN
    prod_petr_diaria = prod_petr_diaria.interpolate(method='polynomial', order=2)

    prod_petr_diaria = prod_petr_diaria.reset_index()

    # colocar o nome nas colunas
    prod_petr_diaria.columns = ['data','prod_petroleo']


    # baixar os dados de preço de petroleo
    response = requests.get(url_preco_petroleo)
    if response.status_code == 200:
        data = response.json()
        petroleo = pd.DataFrame(data['value'])
        petroleo = petroleo[['VALDATA','VALVALOR']].copy()
        petroleo.columns = ['data','cotacao_petroleo']
        petroleo['data'] = petroleo['data'].str[0:10]
        petroleo['data'] = pd.to_datetime(petroleo['data'])
        petroleo['cotacao_petroleo'] = petroleo['cotacao_petroleo'].astype(float)
    else:
        print(f"Error: {response.status_code}")

    dados_diarios_2['datetime'] = pd.to_datetime(dados_diarios_2['datetime'])
    selic['data'] = pd.to_datetime(selic['data'])
    cambio['data'] = pd.to_datetime(cambio['data'])
    prod_petr_diaria['data'] = pd.to_datetime(prod_petr_diaria['data'])
    petroleo['data'] = pd.to_datetime(petroleo['data'])

    dados_diarios_3 = pd.merge(dados_diarios_2, selic, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
    dados_diarios_3 = pd.merge(dados_diarios_3, cambio, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
    dados_diarios_3 = pd.merge(dados_diarios_3, prod_petr_diaria, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
    dados_diarios_3 = pd.merge(dados_diarios_3, petroleo, how='left', left_on='datetime', right_on='data').drop(columns=['data'])

    df = pd.concat([df, dados_diarios_3], ignore_index=True)

    df['prod_petroleo'] = df['prod_petroleo'].ffill()
    df['cotacao_petroleo'] = df['cotacao_petroleo'].ffill()

    df.to_json('database/df_hist.json', orient='records', lines=True)

# Extract just the date
# just_date = dados_diarios.loc['Date'].date()

# print(just_date)

# dados_diarios_2 = dados_diarios.loc.loc[['Date', 'Close']]



# print(dados_diarios)



df = df.rename(columns={'datetime': 'ds', 'close': 'y'})


data_mais_recente = max(df['ds'])



# class ForecastInput(BaseModel):
#     selic: list
#     cambio: list
#     prod_petroleo: list
#     cotacao_petroleo: list

# data = ForecastInput(
#     selic = [14.9],
#     cambio = [5.3457],
#     prod_petroleo = [4031],
#     cotacao_petroleo = [67.88]
# )


df_forecast = pd.DataFrame({"ds": [data_mais_recente+pd.Timedelta(days=1)], "selic": [None], "cambio": [None], "prod_petroleo": [None],
                            "cotacao_petroleo": [None], "y": [None]})

df_2 = pd.concat([df, df_forecast], ignore_index=True)

df_2 = df_2.sort_values('ds').copy()

# df_2 = df.sort_values('ds').copy()

# df_2.drop(columns=['ds'], inplace=True)

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

df_4 = df_3.tail(11).copy()

# print(df_3)

model = load("model/neuralprophet_model.np")
forecast = model.predict(df_4)

yhat = forecast[['ds','yhat1']].iloc[-1]
df_3['y'].iloc[-1] = yhat['yhat1']

target_2 = df_3['y']

target_2_actual = scaler_y.inverse_transform(target_2.values.reshape(-1, 1))

# print(target_2_actual[-1][0])
# print(yhat['ds'])

df_with_forecast = pd.DataFrame([{yhat['ds'], target_2_actual[-1][0]}], columns=['ds','yhat'])
print(df_with_forecast)
# print(target_2_actual)
# print(target.tail())
# print(features.tail())
# print(features.shape)
# print(target.shape)