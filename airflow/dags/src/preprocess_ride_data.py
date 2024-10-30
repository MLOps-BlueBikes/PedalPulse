import pandas as pd
import numpy as np
import math


def data_type_conversion(df):
   df['started_at']  = pd.to_datetime(df['started_at'])
   df['ended_at']  = pd.to_datetime(df['ended_at'])

   df['rideable_type'] = df['rideable_type'].astype('category')
   df['member_casual'] = df['member_casual'].astype('category')

   df['start_station_id'] = df['start_station_id'].astype('str')
   df['end_station_id'] = df['end_station_id'].astype('str') 

   return df

def haversine_distance(lat1, lon1, lat2, lon2):
   # Radius of the Earth in kilometers
   R = 6371.0
   
   # Convert degrees to radians
   lat1_rad = math.radians(lat1)
   lon1_rad = math.radians(lon1)
   lat2_rad = math.radians(lat2)
   lon2_rad = math.radians(lon2)
   
   # Compute differences
   dlat = lat2_rad - lat1_rad
   dlon = lon2_rad - lon1_rad
   
   # Haversine formula
   a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
   c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
   
   # Distance in kilometers
   distance = R * c
   
   return distance

def extract_temporal_features(df):
   df['year'] = df['started_at'].dt.year
   df['month'] = df['started_at'].dt.month
   df['day'] = df['started_at'].dt.day
   df['hour'] = df['started_at'].dt.hour
   df['day_name'] = df['started_at'].dt.day_name()
   df['duration'] = round((df['ended_at'] - df['started_at']) / pd.Timedelta(minutes=1),0)
   df['date'] = df['started_at'].dt.date
   df['distance_km'] = df.apply(lambda row: haversine_distance(row['start_lat'], row['start_lng'], 
                                                           row['end_lat'], row['end_lng']), axis=1)   
   
   return df


'''def preprocess(df, df_weather):
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
    '''

def remove_invalid_data(df):

   df = df[(df['duration'] > 5) & (df['duration'] < 1440)]
   df = df[df['distance_km'] > 0] 

   return df