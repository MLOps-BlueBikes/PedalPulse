
# Weather Data and Ride history Integration ( Yet to apply automation Using API)


It will scrape weather data from Wunderground and integrate it with bike ride data. The purpose is to analyze the relationship between weather conditions and bike usage. The workflow involves:

1. Scraping weather data for specific dates and times.
2. Processing and formatting the scraped data.
3. Merging the weather data with bike ride data based on the closest time.

## Files

- **`web_scraping.ipynb`**: The main notebook for scraping weather data and merging it with the ride data.
- **`ride_data.csv`**: A dataset that contains information on bike rides such as start/end times, stations, coordinates, and user type.
- **`Weather_StationData1.csv`**: Weather Station data Including location of Weather Station

Ride Data => Blue Bike monthly Data from S3 Bucket 

## Prerequisites

Ensure you have the following installed to run the notebook:

- Python 3.x
- Required Python packages:
  ```bash
  pip install pandas beautifulsoup4 selenium

How to Run the Notebook
1. Scraping Weather Data
The notebook uses Selenium and BeautifulSoup to extract historical weather data from Wunderground. You need to define the date range you want to scrape, and the notebook will:

Navigate to Wunderground’s weather history page for each date.
Extract relevant weather information (temperature, humidity, wind, etc.).
Save the data in a structured Pandas DataFrame.
Example scraping function:

python
Copy code
def scrape_wunderground_data(date):
    # Function to scrape weather data for a specific date from Wunderground
    ...
You can modify the date range as needed for your analysis.

2. Merging Weather Data with Ride Data
The weather data is merged with the ride data by matching the closest timestamps. The ride start time (started_at) is compared with the weather data timestamps, and the nearest weather observation is attached to each ride.

Steps:

Convert both started_at (from ride data) and DateTime (from weather data) into a proper datetime format.

Use merge_asof() to perform the time-based merge:

python
Copy code
merged_df = pd.merge_asof(rides_df, weather_df, left_on='started_at', right_on='DateTime', direction='backward')
3. Saving the Merged Data
The merged data (ride data + weather conditions) can be saved into a CSV file for further analysis:

python
Copy code
final_df.to_csv('ride_weather_data.csv', index=False)
Data Format
Ride Data (ride_data.csv)
ride_id: Unique identifier for each bike ride.
rideable_type: Type of bike used (e.g., docked bike).
started_at: Ride start time (to be merged with weather data).
ended_at: Ride end time.
start_station_name, start_station_id, end_station_name, end_station_id: Station details.
start_lat, start_lng, end_lat, end_lng: Latitude and longitude of start and end points.
member_casual: Rider type (member or casual).
Closest Weather Station: Weather station nearest to the ride start.
Weather Station ID: ID of the weather station.
NAME: Name of the station.
Weather Data (weather_data.csv)
Temperature (°F): Temperature readings.
Dew Point: Dew point in Fahrenheit.
Humidity: Humidity percentage.
Wind: Wind direction.
Wind Speed: Wind speed in mph.
Wind Gust: Maximum gust speed.
Pressure: Air pressure in inches.
Precip.: Precipitation in inches.
Condition: Weather condition (e.g., Cloudy).
DateTime: Timestamp of the weather observation (used for merging with ride data).
Running the Notebook
Open web_scraping.ipynb in Jupyter Notebook.
Ensure your WebDriver (e.g., ChromeDriver) is set up correctly.
Run all cells sequentially to:
Scrape weather data for the specified date range.
Merge the weather data with the ride data.
Save the merged data into a CSV file.
Important Notes
Selenium WebDriver: Ensure you have ChromeDriver (or other WebDriver) installed and added to your system's PATH for Selenium scraping.
Scraping Limitations: Wunderground may limit scraping activities (rate limiting, CAPTCHAs). The notebook includes delays to reduce the likelihood of being blocked.
Accuracy of Merging: The merging is based on the closest time match. While this works well for most scenarios, rapidly changing weather conditions may require finer time granularity.
