import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
from math import radians, sin, cos, sqrt, atan2



def add_weather_data(trip_data_dict):

    trip_data=trip_data_dict['df']
    trip_data['started_at'] = pd.to_datetime(trip_data['started_at'])
    trip_data['ended_at'] = pd.to_datetime(trip_data['ended_at'])

    dates_list=sorted(trip_data['started_at'].apply(lambda x: x.date()).unique())
    weather_df=scrape_multiple_days(dates_list)
    rides_df = trip_data.sort_values('started_at')
    weather_df= weather_df.sort_values('DateTime')

    # Using 'started_at' to find the nearest weather data
    merged_df = pd.merge_asof(rides_df, weather_df, left_on='started_at', right_on='DateTime', direction='backward')

    # Step 4: Drop unnecessary columns if needed (like 'DateTime' from weather data)
    merged_df = merged_df.drop(columns=['DateTime'])
    
    merged_df_path=trip_data_dict['path'].split('.')[0]+"-weather-merged.csv"
    merged_df.to_csv(f"data/{merged_df_path}")
    
    return {"df":merged_df,"path":merged_df_path}





# Set up the WebDriver with Chrome options
def set_up_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Ensures Chrome runs without a UI
    options.add_argument('--disable-gpu')  # Disable GPU (helps in some headless cases)
    options.add_argument('--no-sandbox')  # Bypass OS security model
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
    options.add_argument('--remote-debugging-port=9222')  # Open port for debugging

    driver = webdriver.Chrome(options=options)
    return driver

# Function to scrape weather data for a given date
def scrape_wunderground_data(date):
    driver = set_up_driver()

    # URL format based on the provided link
    url = f"https://www.wunderground.com/history/daily/us/ma/east-boston/KBOS/date/{date}"
    driver.get(url)

    # Wait for the page to load completely
    try:
        # Wait until the table with aria-labelledby="History observation" is present
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table[aria-labelledby="History observation"]'))
        )
    except Exception as e:
        print(f"Error: Unable to load table for {date}. {str(e)}")
        driver.quit()
        return None

    # Extract the page source after the table has loaded
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Now, try to find the table in the loaded HTML
    table = soup.find('table', {'aria-labelledby': 'History observation'})

    if not table:
        print(f"No weather data table found for {date}")
        driver.quit()
        return None

    rows = table.find_all('tr')
    weather_data = []

    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        weather_data.append(cols)

    # Define columns for the regular weather data
    columns = ['Time', 'Temperature (°F)', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.', 'Condition']

    df = pd.DataFrame(weather_data, columns=columns[:len(weather_data[0])])  # Adjust based on actual columns in the table

    driver.quit()

    return df


# Function to loop through multiple dates and save data
def scrape_multiple_days(dates_list):
    # Create a date range


    all_data = []

    for date in dates_list:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Scraping data for {date_str}")
        df = scrape_wunderground_data(date_str)

        if df is not None:
            df['Date'] = date_str  # Add the date to each row
            all_data.append(df)

        # Avoid being blocked
        time.sleep(2)

    # Combine all data into a single DataFrame
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

    else:
        print("No data was scraped.")
    df_weather_scrap = final_df.dropna()
    '''
    df_weather_scrap.loc[:, 'DateTime'] = df_weather_scrap['Date'] + ' ' + df_weather_scrap['Time']
    df_weather_scrap.loc[:, 'DateTime'] = pd.to_datetime(df_weather_scrap['DateTime'], format='%Y-%m-%d %I:%M %p')
    '''
    # Use .assign() to create DateTime and convert to datetime format in one step
    df_weather_scrap = df_weather_scrap.assign(
        DateTime=pd.to_datetime(df_weather_scrap['Date'] + ' ' + df_weather_scrap['Time'], format='%Y-%m-%d %I:%M %p')
    )
    df_weather_scrap = df_weather_scrap.drop(columns=['Date', 'Time'])



    return df_weather_scrap

'''
if __name__=="__main__":
    print("Hi")
    df=pd.read_csv("/Users/skc/PedalPulse/airflow/dags/data/202409-bluebikes-tripdata.csv")
    print(f"OLD DATAFRAME SHAPE: {df.shape}")
    new_df=add_weather_data('/Users/skc/PedalPulse/airflow/dags/data/202409-bluebikes-tripdata.csv')
    print(f"New DATAFRAME SHAPE: {new_df.shape}")
'''


