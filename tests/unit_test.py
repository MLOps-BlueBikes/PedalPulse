import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime, timedelta
import re
import pytest


base_url = "https://s3.amazonaws.com/hubway-data"

def monthly_url(logical_date):

    year = logical_date.year
    month = logical_date.month
    return f"{base_url}/{year}{month:02d}-bluebikes-tripdata.zip"

def download_and_extract_data(logical_date):

    while True:
        # Generate the URL for the logical date
        url = monthly_url(logical_date)
        print(f"Attempting to download data from: {url}")

        try:
            
            response = requests.get(url)
            response.raise_for_status()  # error for file not existing

            zip_file = ZipFile(BytesIO(response.content))
            csv_filename = [name for name in zip_file.namelist() if name.endswith('.csv')][0]
            
            with zip_file.open(csv_filename) as csv_file:
                # Step 3: CSV to dataframe
                df = pd.read_csv(csv_file)
                print("Successfully downloaded and extracted data.")
            return df  

        except requests.exceptions.HTTPError:
            print(f"Failed to download data for {logical_date.strftime('%Y-%m')}. Trying the previous month.")
            logical_date = logical_date - timedelta(days=30)  # get previous months data smaple if current data sample is not there


logical_date = datetime.now() #get current date
df = download_and_extract_data(logical_date)


@pytest.fixture
def data():
    # Fixture to provide the downloaded DataFrame 
    return df

#Data Quality
#1 missing values 
def test_missing_values(data): 
    assert data['end_station_name'].isnull().sum() == 0, "Missing values in 'end_station_name'"
    assert data['end_station_id'].isnull().sum() == 0, "Missing values in 'end_station_id'"

#2 check data column miss managment
def test_column_types(data):
    assert pd.api.types.is_string_dtype(data['ride_id']), "string"
    assert pd.api.types.is_string_dtype(data['rideable_type']), "string"
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(data['started_at'], errors='coerce')), "datetime"
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(data['ended_at'], errors='coerce')), "datetime"
    assert pd.api.types.is_float_dtype(data['start_lat']), "float"
    assert pd.api.types.is_float_dtype(data['start_lng']), " float"
    assert pd.api.types.is_float_dtype(data['end_lat']), " float"
    assert pd.api.types.is_float_dtype(data['end_lng']), "float"

#3 check if date format is yyyy-mm-dd hh:mm:sss for started_at and ended_at
def test_date_format(data):

    for date in data['started_at']:
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date), "'started_at' date format is incorrect"
    for date in data['ended_at']:
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date), "'ended_at' date format is incorrect"

#4 duration cant be negative value
def test_duration_calculation(data):
    data['started_at'] = pd.to_datetime(data['started_at'], errors='coerce')
    data['ended_at'] = pd.to_datetime(data['ended_at'], errors='coerce')
    data['duration'] = (data['ended_at'] - data['started_at']).dt.total_seconds()
    assert (data['duration'] >= 0).all(), "Trip duration has negative values"

#5 lat and long shld be in the range -90 to +90 and -180 to +180
def test_invalid_lat_long(data):
    assert ((data['start_lat'] >= -90) & (data['start_lat'] <= 90)).all(), "'start_lat' out of range"
    assert ((data['start_lng'] >= -180) & (data['start_lng'] <= 180)).all(), "'start_lng' out of range"
    assert ((data['end_lat'].dropna() >= -90) & (data['end_lat'].dropna() <= 90)).all(), "'end_lat' out of range"
    assert ((data['end_lng'].dropna() >= -180) & (data['end_lng'].dropna() <= 180)).all(), "'end_lng' out of range"

#6 Ride id shld be unique
def test_unique_ride_id(data):
    assert data['ride_id'].is_unique, "'ride_id' values are not unique"

#7 member_casual column shld have only "member" or "casual"
def test_membership_type(data):
    assert set(data['member_casual'].unique()).issubset({'member', 'casual'}), "'member_casual' has unexpected values"

if __name__ == "__main__":
    pytest.main([__file__])
