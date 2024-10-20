
# Weather Data and Ride history Integration ( Yet to apply automation Using API)


It will scrape weather data from Wunderground and integrate it with bike ride data. The purpose is to analyze the relationship between weather conditions and bike usage. The workflow involves:

1. Scraping weather data for specific dates and times.
2. Processing and formatting the scraped data.
3. Merging the weather data with bike ride data based on the closest time.

## Files

- **`web_scraping.ipynb`**: The main Jupyter notebook for scraping weather data and merging it with the ride data.
- **`ride_data.csv`**: A dataset that contains information on bike rides such as start/end times, stations, coordinates, and user type.
- **`weather_data.csv`**: Weather data scraped from Wunderground for relevant dates, including temperature, humidity, wind conditions, and other metrics.

## Prerequisites

Ensure you have the following installed to run the notebook:

- Python 3.x
- Required Python packages:
  ```bash
  pip install pandas beautifulsoup4 selenium

