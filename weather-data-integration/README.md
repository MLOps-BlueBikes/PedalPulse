
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

Workflow
--------

### 1\. Saving the Merged Data

Once the merge is complete, the resulting dataset, which now includes weather conditions for each bike ride, can be saved to a CSV file for further analysis:


### 2\. Data Format

#### Ride Data (ride\_data.csv)

*   **ride\_id**: Unique identifier for each bike ride.
    
*   **rideable\_type**: Type of bike used (e.g., docked bike).
    
*   **started\_at**: Start time of the ride (to be merged with weather data).
    
*   **ended\_at**: End time of the ride.
    
*   **start\_station\_name, start\_station\_id**: Information about the start station.
    
*   **end\_station\_name, end\_station\_id**: Information about the end station.
    
*   **start\_lat, start\_lng**: Latitude and longitude of the start point.
    
*   **end\_lat, end\_lng**: Latitude and longitude of the end point.
    
*   **member\_casual**: Type of rider (member or casual).
    
*   **Closest Weather Station**: Name of the nearest weather station.
    
*   **Weather Station ID**: ID of the weather station.
    

#### Weather Data (weather\_data.csv)

*   **Temperature (°F)**: Temperature reading in Fahrenheit.
    
*   **Dew Point**: Dew point temperature.
    
*   **Humidity**: Humidity percentage.
    
*   **Wind**: Wind direction.
    
*   **Wind Speed**: Wind speed in mph.
    
*   **Wind Gust**: Maximum wind gust speed.
    
*   **Pressure**: Air pressure in inches.
    
*   **Precip.**: Precipitation in inches.
    
*   **Condition**: Weather condition (e.g., Cloudy, Rainy).
    
*   **DateTime**: Timestamp of the weather observation (used for merging).
    

### 3\. Running the Notebook

Follow these steps to run the Jupyter notebook:

1.  Open the web\_scraping.ipynb notebook in Jupyter.
    
2.  Ensure you have Selenium WebDriver (e.g., ChromeDriver) installed and properly configured for web scraping.
    
3.  Run all cells to:
    
    *   Scrape the weather data for the specified date range.
        
    *   Merge the weather data with the bike ride data based on the closest matching timestamps.
        
    *   Save the final merged dataset to a CSV file.
        

### 4\. Important Notes

*   **Selenium WebDriver**: Ensure that ChromeDriver (or another WebDriver for your browser) is installed and added to your system’s PATH. Selenium is required for web scraping.
    
*   **Scraping Limitations**: Be aware that Wunderground may have rate limits or CAPTCHAs. The notebook includes sleep delays between requests to reduce the risk of being blocked.
    
*   **Merging Accuracy**: The merge is performed using the nearest available weather timestamp before the ride start time. In cases of rapidly changing weather, there may be slight mismatches in the weather conditions at the exact ride start time.
    

