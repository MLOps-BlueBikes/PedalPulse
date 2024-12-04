import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime, timedelta
import re
import pytest

base_url = "https://s3.amazonaws.com/hubway-data"


def monthly_url(logical_date):
    """Generate the monthly URL for the data file based on the logical date."""
    year = logical_date.year
    month = logical_date.month
    return f"{base_url}/{year}{month:02d}-bluebikes-tripdata.zip"


def download_and_extract_data(logical_date):
    """Download and extract the CSV file for the specified logical date."""
    while True:
        url = monthly_url(logical_date)
        print(f"Attempting to download data from: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            zip_file = ZipFile(BytesIO(response.content))
            csv_filename = [
                name for name in zip_file.namelist() if name.endswith(".csv")
            ][0]

            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                print("Successfully downloaded and extracted data.")
            return df

        except requests.exceptions.HTTPError:
            print(
                f"Failed to download data for {logical_date.strftime('%Y-%m')}."
                " Trying the previous month."
            )
            logical_date -= timedelta(days=30)


logical_date = datetime.now()
df = download_and_extract_data(logical_date)


@pytest.fixture
def data():
    """Fixture to provide the downloaded DataFrame."""
    return df


# Data Quality Tests


def test_missing_values(data):
    """Test for missing values in important columns."""
    max_allowed_missing = 1000
    missing_count = data["end_station_name"].isnull().sum()
    assert (
        missing_count <= max_allowed_missing
    ), f"Too many missing values in 'end_station_name'. Found {missing_count}."


def test_column_types(data):
    """Test column data types for consistency."""
    assert pd.api.types.is_string_dtype(data["ride_id"])
    assert pd.api.types.is_string_dtype(data["rideable_type"])
    assert pd.api.types.is_datetime64_any_dtype(
        pd.to_datetime(data["started_at"], errors="coerce")
    )
    assert pd.api.types.is_datetime64_any_dtype(
        pd.to_datetime(data["ended_at"], errors="coerce")
    )
    assert pd.api.types.is_float_dtype(data["start_lat"])
    assert pd.api.types.is_float_dtype(data["start_lng"])
    assert pd.api.types.is_float_dtype(data["end_lat"])
    assert pd.api.types.is_float_dtype(data["end_lng"])


def test_date_format(data):
    """Test that dates follow the correct format (YYYY-MM-DD HH:MM:SS)."""
    date_regex = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    for date in data["started_at"]:
        assert re.match(date_regex, date)
    for date in data["ended_at"]:
        assert re.match(date_regex, date)


def test_duration_calculation(data):
    """Test that trip duration can be calculated and is non-negative."""
    data["started_at"] = pd.to_datetime(data["started_at"], errors="coerce")
    data["ended_at"] = pd.to_datetime(data["ended_at"], errors="coerce")
    data["duration"] = (data["ended_at"] - data["started_at"]).dt.total_seconds()
    assert (data["duration"] >= 0).all(), "Trip duration has negative values"


def test_invalid_lat_long(data):
    """Test that latitude and longitude are within valid ranges."""
    assert ((data["start_lat"] >= -90) & (data["start_lat"] <= 90)).all()
    assert ((data["start_lng"] >= -180) & (data["start_lng"] <= 180)).all()
    assert ((data["end_lat"].dropna() >= -90) & (data["end_lat"].dropna() <= 90)).all()
    assert (
        (data["end_lng"].dropna() >= -180) & (data["end_lng"].dropna() <= 180)
    ).all()


def test_unique_ride_id(data):
    """Test that ride_id values are unique."""
    assert data["ride_id"].is_unique, "'ride_id' values are not unique"


def test_membership_type(data):
    """Test that member_casual column only contains 'member' or 'casual' values."""
    valid_types = {"member", "casual"}
    assert set(data["member_casual"].unique()).issubset(valid_types)


if __name__ == "__main__":
    pytest.main([__file__])
