


def pull_weather_and_trip_data(**kwargs):
    date_obj = datetime.fromisoformat(ds) - timedelta(days=30)
    yr_mnth = date_obj.strftime('%Y%m')
    
    weather_df=pd.read_csv('/opt/airflow/dags/data/preprocessed_tripdata/'+f'{yr_mnth}-weather-data.csv')
    tripdata_df=pd.read_csv('/opt/airflow/dags/data/preprocessed_tripdata/'+f'preprocessed-{yr_mnth}-bluebikes-tripdata.csv')

    # Assuming the two DataFrames are combined_df_bike and combined_df_weather
    merged_df = pd.merge(tripdata_df, weather_df, on='ride_id', how='inner')
    merged_df = merged_df.rename(columns={
        'rideable_type_x': 'rideable_type',
        'started_at_x': 'started_at',
        'ended_at_x': 'ended_at',
        'start_station_name_x': 'start_station_name',
        'start_station_id_x': 'start_station_id',
        'end_station_name_x': 'end_station_name',
        'end_station_id_x': 'end_station_id',
        'start_lat_x': 'start_lat',
        'start_lng_x': 'start_lng',
        'end_lat_x': 'end_lat',
        'end_lng_x': 'end_lng',
        'member_casual_x': 'member_casual',
        # Continue for remaining columns if necessary
    })

    # Specify final column order
    final_columns = [
        'ride_id', 'rideable_type', 'started_at', 'ended_at',
        'start_station_name', 'start_station_id', 'end_station_name',
        'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng',
        'member_casual', 'year', 'month', 'day', 'hour', 'day_name', 'duration',
        'distance_km', 'Closest Weather Station', 'Weather Station ID',
        'NAME', 'Temperature (°F)', 'Dew Point', 'Humidity', 'Wind',
        'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.', 'Condition',
        'DateTime'
    ]    

    final_df = merged_df[final_columns]

    # Group by 'started_at' and 'start_station_name', count rides, and reset index
    bike_undocked = final_df.groupby(['started_at', 'start_station_name']).size().reset_index(name='BikeUndocked')

    # Merge the result back into final_df
    final_df = pd.merge(final_df, bike_undocked, on=['started_at', 'start_station_name'], how='left')

    # List of columns to drop
    columns_to_keep = ['started_at','start_station_name', 'start_station_id','month', 'hour', 'day_name','duration',
        'distance_km','Temperature (°F)','Humidity','Wind Speed','Precip.', 'Condition','BikeUndocked']

    # Select only the columns to keep
    data_cleaned = final_df[columns_to_keep]

    #data_cleaned.to_csv('/opt/airflow/dags/data/merged/f'{yr_mnth}-merged.csv', index=False)
                        
    data_cleaned['started_at'] = pd.to_datetime(data_cleaned['started_at'], format='mixed', errors='coerce')

    # Set 'started_at' as the index for resampling
    data_cleaned.set_index('started_at', inplace=True)


    # If you want to keep non-numeric columns, you can use a custom aggregation function
    hourly_station_data = data_cleaned.groupby('start_station_name').resample('H').agg({
        'month': 'first',                # Keep first value (since it's constant for each hour)
        'hour': 'first',                 # Same as above
        'day_name': 'first',             # Same
        'duration': 'sum',               # Sum durations for the hour
        'distance_km': 'sum',            # Sum distances for the hour
        'Temperature (°F)': 'mean',      # Average temperature
        'Humidity': 'mean',              # Average humidity
        'Wind Speed': 'mean',            # Average wind speed
        'Precip.': 'sum',                # Total precipitation for the hour
        'Condition': 'first',            # Keep first condition as representative
        'BikeUndocked': 'sum'            # Sum undocked bikes
    })

    # Reset the index to make it easier to work with
    hourly_station_data = hourly_station_data.reset_index()
    hourly_station_data.to_csv('/opt/airflow/dags/data/merged/'+f'{yr_mnth}-merged.csv', index=False)





