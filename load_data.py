import pandas as pd
import re
import requests

cota2025 = pd.read_csv("C:/Users/Kleber/Downloads/COTAHIST_A2025.TXT")

# funcao para extrair os caracteres até aparecer o primeiro espaço
def extract_from_pos_until_space(s, start=12):
    s = s[start:]
    match = re.match(r'^\S+', s)
    return match.group(0) if match else ''

# funcao para gerar as colunas com as informações necessárias
def org_base(base, tickers, ini_ticker=12):

  base.loc[:,'datetime'] = base.iloc[:,0].str[2:6] + '-' + base.iloc[:,0].str[6:8]  + '-' + base.iloc[:,0].str[8:10]
  base.loc[:,'ticker'] = base.iloc[:,0].apply(lambda x: extract_from_pos_until_space(x, start=ini_ticker))
  base = base[base['ticker'].isin(tickers)].copy()

  base.loc[:,'open'] = base.iloc[:,0].str[56:69]
  base.loc[:,'open'] = pd.to_numeric(base['open'])/100

  base.loc[:,'high'] = base.iloc[:,0].str[69:82]
  base.loc[:,'high'] = pd.to_numeric(base['high'])/100

  base.loc[:,'low'] = base.iloc[:,0].str[82:95]
  base.loc[:,'low'] = pd.to_numeric(base['low'])/100

  base.loc[:,'close'] = base.iloc[:,0].str[108:121]
  base.loc[:,'close'] = pd.to_numeric(base['close'])/100

  base.loc[:,'volume'] = base.iloc[:,0].str[170:188]
  base.loc[:,'volume'] = pd.to_numeric(base['volume'])/100

  base = base.drop(columns=[base.columns[0]])

  return base

df_2025_alt = org_base(cota2025, ['PETR4'])

df_2025_alt['datetime'] = pd.to_datetime(df_2025_alt['datetime'])

df_2025_alt.drop(columns=['ticker','open','high','low','volume'], inplace=True)

# baixar os dados ==========================================

# baixar os dados de Selic, cambio R$/USD e preço do petróleo
url_selic_2025 = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json&dataInicial=01/01/2025'
url_cambio_2025 = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=json&dataInicial=01/01/2025'
url_preco_petroleo = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='EIA366_PBRENT366')"
url_prod_petroleo = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1391/dados?formato=json'

# baixar os dados de selic de 2025
response = requests.get(url_selic_2025)
if response.status_code == 200:
    data = response.json()
    selic_2025 = pd.DataFrame(data)
    selic_2025['data'] = pd.to_datetime(selic_2025['data'], format='%d/%m/%Y')
    selic_2025['selic'] = selic_2025['valor'].astype(float)
    selic_2025.drop(columns=['valor'], inplace=True)
else:
    print(f"Error: {response.status_code}")

# baixar os dados de cambio de 2025
response = requests.get(url_cambio_2025)
if response.status_code == 200:
    data = response.json()
    cambio_2025 = pd.DataFrame(data)
    cambio_2025['data'] = pd.to_datetime(cambio_2025['data'], format='%d/%m/%Y')
    cambio_2025['cambio'] = cambio_2025['valor'].astype(float)
    cambio_2025.drop(columns=['valor'], inplace=True)
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

# fazer merge dos dados ==========================================

df_2025_alt_2 = pd.merge(df_2025_alt, selic_2025, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
df_2025_alt_2 = pd.merge(df_2025_alt_2, cambio_2025, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
df_2025_alt_2 = pd.merge(df_2025_alt_2, prod_petr_diaria, how='left', left_on='datetime', right_on='data').drop(columns=['data'])
df_2025_alt_2 = pd.merge(df_2025_alt_2, petroleo, how='left', left_on='datetime', right_on='data').drop(columns=['data'])


df_2025_alt_2['prod_petroleo'] = df_2025_alt_2['prod_petroleo'].ffill()
df_2025_alt_2['cotacao_petroleo'] = df_2025_alt_2['cotacao_petroleo'].ffill()

# carregar a base de dados históricos e concatenar os dados ==========================================

df_hist = pd.read_json("database/df_hist.json", orient='records', lines=True)

df_hist = df_hist[df_hist['datetime']<=pd.to_datetime('2025-09-15')]

df_2025_alt_2 = df_2025_alt_2[df_2025_alt_2['datetime']>(max(df_hist['datetime']))]

df_2025_alt_2 = df_2025_alt_2.sort_values('datetime').copy()

df_hist_2 = pd.concat([df_hist, df_2025_alt_2], ignore_index=True)

df_hist_2.dropna(inplace=True)

df_hist_2.to_json('database/df_hist.json', orient='records', lines=True)
