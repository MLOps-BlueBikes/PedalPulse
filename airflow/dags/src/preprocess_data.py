import pandas as pd
import numpy as np
#from src.ingest_data import load_blue_bikes_data, load_weather_data

def preprocess(df, df_weather):
    df['started_at']  = pd.to_datetime(df['started_at'])
    df['ended_at']  = pd.to_datetime(df['ended_at'])

    df['year'] = df['started_at'].dt.year
    df['month'] = df['started_at'].dt.month
    df['day'] = df['started_at'].dt.day
    df['hour'] = df['started_at'].dt.hour
    df['day_name'] = df['started_at'].dt.day_name()
    df['duration'] = round((df['ended_at'] - df['started_at']) / pd.Timedelta(minutes=1),0)
    df['date'] = df['started_at'].dt.date

    df_copy = df.copy()

    df.drop(columns=['ride_id','started_at','ended_at',
       'start_lat', 'start_lng', 'end_lat', 'end_lng'], inplace = True)
    
    df = df.dropna(subset=['start_station_id', 'end_station_id'])

    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    df['date'] = pd.to_datetime(df['date'])

    merged_df = pd.merge(df, df_weather, left_on='date', right_on = 'Date',how='inner')

    return merged_df