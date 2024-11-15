import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import logging
import calendar
from datetime import date

from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Set up Selenium WebDriver
def set_up_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Remote(
        command_executor='http://selenium:4444/wd/hub',
        options=options
    )
    return driver


# Scrape weather data for a specific date
def scrape_wunderground_data(date):
    driver = set_up_driver()
    url = (
        f"https://www.wunderground.com/history/daily/us/ma/east-boston/KBOS/date/{date}"
    )
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'table[aria-labelledby="History observation"]')
            )
        )
    except Exception as e:
        logging.error(f"Error: Unable to load table for {date}. {str(e)}")
        driver.quit()
        return None

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", {"aria-labelledby": "History observation"})

    if not table:
        logging.error(f"No weather data table found for {date}")
        driver.quit()
        return None

    rows = table.find_all("tr")
    weather_data = [
        [col.text.strip() for col in row.find_all("td")] for row in rows[1:]
    ]
    columns = [
        "Time",
        "Temperature (°F)",
        "Dew Point",
        "Humidity",
        "Wind",
        "Wind Speed",
        "Wind Gust",
        "Pressure",
        "Precip.",
        "Condition",
    ]
    df = pd.DataFrame(weather_data, columns=columns[: len(weather_data[0])])
    driver.quit()
    return df

# Load Bluebikes trip data with error handling
def read_bike_trip_data(month):
    try:
        # Read from GCP
        file_name = f"preprocessed_{month}-bluebikes-tripdata.csv"

        client = storage.Client()
        # Get the bucket
        bucket_name = 'trip_data_bucket_testing'
        bucket = client.bucket(bucket_name)
        logging.info(f"Trying to fetch gcs bucket:{bucket_name}")
        # Get the blob
        blob_name = f'Preprocessed_data/{file_name}'
        blob = bucket.blob(blob_name)
        # Download the contents
        destination_file_name = f"/opt/airflow/dags/data/{file_name}"
        blob.download_to_filename(destination_file_name)
        logging.info(f"Downloaded {blob_name} from GCS to {destination_file_name}")

        return destination_file_name

    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

# Get date range for month
def get_month_start_end(month, year):
    """Get start and end date of given month."""
    # First day of the current month
    start_date = date(year, month, 1)
    # Last day of the current month
    _, last_day = calendar.monthrange(year, month)
    end_date = date(year, month, last_day)
    
    return start_date, end_date

# Scrape data for multiple dates and save as CSV
def scrape_multiple_days(file_name):
    output_files = []
    output_files.append(file_name)

    # Extract the year and month part from the file name (e.g., '202401')
    year_month = file_name.split('_')[1].split('-')[0]  # '202401'
    # Extract the year and month components
    year = int(year_month[:4])  # First 4 digits: year
    month = int(year_month[4:])  # Last 2 digits: month

    start_date, end_date = get_month_start_end(month, year)
    date_range = pd.date_range(start=start_date, end=end_date)
    all_data = []

    logging.info("Starting weather data scraping...")
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        logging.info(f"Scraping data for {date_str}")
        df = scrape_wunderground_data(date_str)
        if df is not None:
            df["Date"] = date_str
            all_data.append(df)
        time.sleep(2)  # Avoid being blocked

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_path = f"/opt/airflow/dags/data/weather_data.csv"
        final_df.to_csv(output_path, index=False)
        logging.info(f"Weather data saved to {output_path}")
        output_files.append(output_path)
        return output_files
    else:
        logging.warning("No data scraped.")
        return None

# Match rides with closest weather data by timestamp
def match_rides_with_weather(data_files):
    
    rides_file, weather_file = data_files

    rides_df = pd.read_csv(rides_file)
    weather_df = pd.read_csv(weather_file)

    rides_df["started_at"] = pd.to_datetime(rides_df["started_at"])
    weather_df["DateTime"] = pd.to_datetime(
        weather_df["Date"] + " " + weather_df["Time"], format="%Y-%m-%d %I:%M %p"
    )
    closest_weather_data = []

    for idx, ride_row in rides_df.iterrows():
        time_diffs = abs(weather_df["DateTime"] - ride_row["started_at"])
        nearest_index = time_diffs.idxmin()
        closest_weather = weather_df.iloc[nearest_index]

        combined_data = ride_row.to_dict()
        combined_data.update(closest_weather.to_dict())
        closest_weather_data.append(combined_data)

    merged_weather = pd.DataFrame(closest_weather_data)

    # Convert columns with numeric values
    for column in [
        "Temperature (°F)",
        "Dew Point",
        "Humidity",
        "Wind Speed",
        "Pressure",
        "Precip.",
    ]:
        merged_weather[column] = (
            merged_weather[column].str.extract("([0-9.]+)").astype(float)
        )
    merged_weather["Wind Gust"] = (
        merged_weather["Wind Gust"].str.extract("([0-9.]+)").astype(float)
    )

    # Drop columns 'Date' and 'DateTime'
    merged_weather.drop(columns=['Date', 'DateTime'], inplace=True)

    # Add 'bike_undocked' column for all records
    merged_weather['bike_undocked'] = 1

    #merged_weather.to_csv("weather_trip_history_merged.csv", index=False)
    output_path = f"/opt/airflow/dags/data/weather_trip_history_merged.csv"
    merged_weather.to_csv(output_path, index=False)
    logging.info(f"Weather-trip data merged and saved to {output_path}.")
    return output_path

# Upload file to GCP bucket
def upload_to_gcp(file_path, bucket_name, month):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    file_name = f'bike_weather_{month}.csv'
    blob_name = f'merged_data/{file_name}'
    blob = bucket.blob(blob_name)

    retry_count = 0
    max_retries = 10
    while retry_count < max_retries:
        try:
            blob.upload_from_filename(file_path)
            logging.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
            break
        except Exception as e:
            retry_count += 1
            time.sleep(2**retry_count)  # Exponential backoff
            if retry_count == max_retries:
                logging.error(
                    f"Failed to upload {file_path} after {max_retries} retries"
                )
            logging.error(f"Error uploading {file_path}: {e}")
            raise AirflowFailException(f"Failed to upload {file_path}:{e}")
