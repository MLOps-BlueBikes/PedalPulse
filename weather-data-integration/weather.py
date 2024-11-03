import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate Haversine distance between two lat-long points
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371.0  # Earth radius in kilometers
    return R * c

# Set up Selenium WebDriver
def set_up_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver

# Scrape weather data for a specific date
def scrape_wunderground_data(date):
    driver = set_up_driver()
    url = f"https://www.wunderground.com/history/daily/us/ma/east-boston/KBOS/date/{date}"
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table[aria-labelledby="History observation"]'))
        )
    except Exception as e:
        logging.error(f"Error: Unable to load table for {date}. {str(e)}")
        driver.quit()
        return None

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {'aria-labelledby': 'History observation'})

    if not table:
        logging.error(f"No weather data table found for {date}")
        driver.quit()
        return None

    rows = table.find_all('tr')
    weather_data = [[col.text.strip() for col in row.find_all('td')] for row in rows[1:]]
    columns = ['Time', 'Temperature (°F)', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.', 'Condition']
    df = pd.DataFrame(weather_data, columns=columns[:len(weather_data[0])])
    driver.quit()
    return df

# Scrape data for multiple dates and save as CSV
def scrape_multiple_days(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    all_data = []

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        logging.info(f"Scraping data for {date_str}")
        df = scrape_wunderground_data(date_str)
        if df is not None:
            df['Date'] = date_str
            all_data.append(df)
        time.sleep(2)  # Avoid being blocked

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv('weather_data.csv', index=False)
        logging.info("Weather data saved to 'weather_data.csv'")
    else:
        logging.warning("No data scraped.")

# Load Bluebikes trip data with error handling
def read_bike_trip_data(file_name):
    try:
        df = pd.read_csv(file_name, on_bad_lines='skip')
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['ended_at'] = pd.to_datetime(df['ended_at'])
        return df
    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

# Process Bluebikes and weather data for specified months
def process_bike_trip_weather_data():
    months = ['202401', '202402', '202403', '202404', '202405', '202406', '202407', '202408', '202409']
    for month in months:
        logging.info(f"Processing data for {month}...")
        file_name = f"{month}-bluebikes-tripdata.csv"
        df = read_bike_trip_data(file_name)

        if df is not None:
            unique_station_names = df['start_station_name'].unique()
            unique_stations_with_ids = df[df['start_station_name'].isin(unique_station_names)]

            station_name_id_mapping = unique_stations_with_ids[['start_station_name', 'start_station_id', 'start_lat', 'start_lng']]
            station_name_id_mapping.to_csv(f'station_name_id_mapping_{month}.csv', index=False)
            logging.info(f"Station data for {month} processed successfully.")
        else:
            logging.warning(f"Skipping processing for {month} due to data loading issues.")

# Match rides with closest weather data by timestamp
def match_rides_with_weather(rides_df, weather_df):
    rides_df['started_at'] = pd.to_datetime(rides_df['started_at'])
    weather_df['DateTime'] = pd.to_datetime(weather_df['Date'] + ' ' + weather_df['Time'], format='%Y-%m-%d %I:%M %p')
    closest_weather_data = []

    for idx, ride_row in rides_df.iterrows():
        time_diffs = abs(weather_df['DateTime'] - ride_row['started_at'])
        nearest_index = time_diffs.idxmin()
        closest_weather = weather_df.iloc[nearest_index]

        combined_data = ride_row.to_dict()
        combined_data.update(closest_weather.to_dict())
        closest_weather_data.append(combined_data)

    merged_weather = pd.DataFrame(closest_weather_data)

    # Convert columns with numeric values
    for column in ['Temperature (°F)', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Precip.']:
        merged_weather[column] = merged_weather[column].str.extract('([0-9.]+)').astype(float)
    merged_weather['Wind Gust'] = merged_weather['Wind Gust'].str.extract('([0-9.]+)').astype(float)

    merged_weather.to_csv('weather_trip_history_merged.csv', index=False)
    logging.info("Weather-trip data merged and saved to 'weather_trip_history_merged.csv'.")

# Main function to call and coordinate the above functions
def main():
    # Define date range for weather data scraping
    start_date = "2024-02-01"
    end_date = "2024-02-02"

    # Step 1: Scrape weather data
    logging.info("Starting weather data scraping...")
    scrape_multiple_days(start_date, end_date)

    # Step 2: Process Bluebikes trip data
    logging.info("Processing Bluebikes trip data for each month...")
    process_bike_trip_weather_data()

    # Step 3: Load data for matching
    logging.info("Loading scraped weather data and trip data for matching...")
    
    try:
        weather_df = pd.read_csv('weather_data.csv')
        bike_trip_df = pd.read_csv('202401-bluebikes-tripdata.csv')
        
        # Step 4: Match rides with weather data
        logging.info("Matching rides with closest weather data by timestamp...")
        match_rides_with_weather(bike_trip_df, weather_df)
    except Exception as e:
        logging.error(f"Error during data loading or processing: {e}")

if __name__ == "__main__":
    main()
