# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:54:20 2023

@author: Peri
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from math import sqrt

from prophet import Prophet
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

myfavouritenumber = 15
seed = myfavouritenumber
np.random.seed(seed)

df = pd.read_csv("D:/CURSO JUNTA ANDALUCIA/EXPERTO INTELIGENCIA ARTIFICIAL/MODULO 5/PROYECTO FINAL/DATA/AXISBANK.csv")
df.set_index("Date", drop=False, inplace=True)
df.head()

df.VWAP.plot(figsize=(14, 7))

df.reset_index (drop=True, inplace=True)
lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]
    
df.fillna(df.mean(), inplace=True)

df.set_index("Date", drop=False, inplace=True)
df.head()


#Creamos tres columna con el mes, la semana, el dia y el dia de semana

df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
df["month"] = df.Date.dt.month
df["week"] = df.Date.dt.week
df["day"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.dayofweek
df.head()

#Dividimos los datos en train y validacion train al 31 de diciembre 2020
#y validaion 1 de enero 2020 al 30 de abril 2021

df_train = df[df.Date < "2020"]
df_valid = df[df.Date >= "2020"]



exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", 
                      "Low_std_lag3", "Volume_mean_lag3", "Volume_std_lag3", 
                      "Turnover_mean_lag3", "Turnover_std_lag3", 
                      "Trades_mean_lag3", "Trades_std_lag3", "High_mean_lag7", 
                      "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",
                      "Volume_mean_lag7", "Volume_std_lag7", 
                      "Turnover_mean_lag7", "Turnover_std_lag7",
                      "Trades_mean_lag7", "Trades_std_lag7", "High_mean_lag30",
                      "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",
                      "Volume_mean_lag30", "Volume_std_lag30", 
                      "Turnover_mean_lag30", "Turnover_std_lag30", 
                      "Trades_mean_lag30", "Trades_std_lag30",
                      "month", "week", "day", "day_of_week"]



#Ahora vamos a utilizar ARIMAX que comunmente se denomina ARIMA es un modelo
#que pronostica el valor futuro en función de sus valores pasados y los 
#errores de pronostricos pasado. Dicho modelo necesita datos de entrada
#que se divide en tres partes AR, MA y I 

model = auto_arima(df_train.VWAP, exogenous=df_train[exogenous_features], 
                   trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.VWAP, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast

#df_valid[["VWAP", "Forecast_ARIMAX"]].plot(figsize=(14, 7))

#print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))
#print("MAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))



#Modelo siguiente Prophet es un procedimiento para pronosticar de datos
#de series temporales basado en un modelo aditivo en las tendencias no lineales
#se ajustan a la estacionalidad anual, semanal y diaria.
#Creado por Facebook 
 

model_fbp = Prophet()
for feature in exogenous_features:
    model_fbp.add_regressor(feature)

model_fbp.fit(df_train[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds", "VWAP": "y"}))

forecast = model_fbp.predict(df_valid[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds"}))
df_valid["Forecast_Prophet"] = forecast.yhat.values

model_fbp.plot_components(forecast)

df_valid[["VWAP", "Forecast_ARIMAX", "Forecast_Prophet"]].plot(figsize=(14, 7))

#print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))
print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_Prophet)))
#print("MAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))
print("MAE of Prophet:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_Prophet))


#Modelo LightGBM 

params = {"objective": "regression"}

dtrain = lgb.Dataset(df_train[exogenous_features], label=df_train.VWAP.values)
dvalid = lgb.Dataset(df_valid[exogenous_features])

model_lgb = lgb.train(params, train_set=dtrain)

forecast = model_lgb.predict(df_valid[exogenous_features])
df_valid["Forecast_LightGBM"] = forecast


df_valid[["VWAP", "Forecast_ARIMAX", "Forecast_Prophet", "Forecast_LightGBM"]].plot(figsize=(14, 7))


#print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))
print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_Prophet)))
print("RMSE of LightGBM:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_LightGBM)))
#print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))
print("MAE of Prophet:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_Prophet))
print("MAE of LightGBM:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_LightGBM))





