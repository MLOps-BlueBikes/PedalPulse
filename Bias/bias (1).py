import pandas as pd
import math

df1 = pd.read_csv("weather_trip_history202401.csv")
df2 = pd.read_csv("weather_trip_history202403.csv")
df3 = pd.read_csv("weather_trip_history202405.csv")

# Append the dataframes
combined_df = pd.concat([df1, df2, df3], ignore_index=True)
# Group by 'started_at' and 'start_station_name', count rides, and reset index
bike_undocked = (
    combined_df.groupby(["started_at", "start_station_name"])
    .size()
    .reset_index(name="BikeUndocked")
)

# # Merge the result back into final_df
final_df = pd.merge(
    combined_df, bike_undocked, on=["started_at", "start_station_name"], how="left"
)


def data_type_conversion(df):
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["rideable_type"] = df["rideable_type"].astype("category")
    df["member_casual"] = df["member_casual"].astype("category")
    df["start_station_id"] = df["start_station_id"].astype("str")
    df["end_station_id"] = df["end_station_id"].astype("str")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_temporal_features(df):
    df["year"] = df["started_at"].dt.year
    df["month"] = df["started_at"].dt.month
    df["day"] = df["started_at"].dt.day
    df["hour"] = df["started_at"].dt.hour
    df["day_name"] = df["started_at"].dt.day_name()
    df["duration"] = round(
        (df["ended_at"] - df["started_at"]) / pd.Timedelta(minutes=1), 0
    )
    df["distance_km"] = df.apply(
        lambda row: haversine_distance(
            row["start_lat"], row["start_lng"], row["end_lat"], row["end_lng"]
        ),
        axis=1,
    )
    return df


def remove_invalid_data(df):
    df = df[(df["duration"] > 5) & (df["duration"] < 1440)]
    df = df[df["distance_km"] > 0]
    return df


final_df = data_type_conversion(final_df)
final_df = extract_temporal_features(final_df)
final_df = remove_invalid_data(final_df)

# List of columns to drop
columns_to_keep = [
    "started_at",
    "start_station_name",
    "start_station_id",
    "month",
    "hour",
    "day_name",
    "duration",
    "distance_km",
    "Temperature (°F)",
    "Humidity",
    "Wind Speed",
    "Precip.",
    "Condition",
    "BikeUndocked",
    "member_casual",
    "rideable_type",
]

# Select only the columns to keep
data_cleaned = final_df[columns_to_keep]

# Display the updated DataFrame
print(data_cleaned.columns)


import pandas as pd

data_cleaned["started_at"] = pd.to_datetime(
    data_cleaned["started_at"], format="mixed", errors="coerce"
)

# Set 'started_at' as the index for resampling
data_cleaned.set_index("started_at", inplace=True)


# If you want to keep non-numeric columns, you can use a custom aggregation function
hourly_station_data = (
    data_cleaned.groupby("start_station_name")
    .resample("H")
    .agg(
        {
            "month": "first",  # Keep first value (since it's constant for each hour)
            "hour": "first",  # Same as above
            "day_name": "first",  # Same
            "duration": "sum",  # Sum durations for the hour
            "distance_km": "sum",  # Sum distances for the hour
            "Temperature (°F)": "mean",  # Average temperature
            "Humidity": "mean",  # Average humidity
            "Wind Speed": "mean",  # Average wind speed
            "Precip.": "sum",  # Total precipitation for the hour
            "Condition": "first",  # Keep first condition as representative
            "BikeUndocked": "sum",  # Sum undocked bikes
        }
    )
)

# Reset the index to make it easier to work with
hourly_station_data = hourly_station_data.reset_index()

# Display the resampled data
print(hourly_station_data.head())

# One-hot encode 'day_name' and 'Condition'
encoded_data = pd.get_dummies(
    hourly_station_data, columns=["day_name", "Condition"], drop_first=True
)

X = encoded_data.drop("BikeUndocked", axis=1)  # Features
y = encoded_data["BikeUndocked"]  # Target
# Frequency encoding This approach encodes each category by its frequency of occurrence in the dataset.
X = X.drop(columns=["started_at"])
station_freq = X["start_station_name"].value_counts().to_dict()
X["start_station_name"] = X["start_station_name"].map(station_freq)
X = X.fillna(0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# import mlflow
# import mlflow.sklearn
import matplotlib.pyplot as plt

# Define the pipeline with StandardScaler and LinearRegression
pipe_lr = Pipeline([("scl", StandardScaler()), ("reg", LinearRegression())])

# Define grid search parameters
param_grid_lr = {"reg__fit_intercept": [True, False], "reg__positive": [True, False]}

# GridSearchCV with cross-validation
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)

# Start an MLflow run
with mlflow.start_run(run_name="Linear Regression with GridSearch"):
    grid_lr.fit(X_train, y_train)

    # Log the best parameters
    best_params = grid_lr.best_params_
    mlflow.log_params(best_params)

    # Evaluate on test data and log metrics
    y_pred = grid_lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # Calculate R² Score
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Log the best model
    mlflow.sklearn.log_model(grid_lr.best_estimator_, "linear_regression_model")

    print("Best parameters: {}".format(best_params))
    print("Mean Squared Error: {}".format(mse))
    print("R² Score: {}".format(r2))

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.grid(True)
    plt.show()

# Import necessary libraries
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Calculate station frequency
station_freq = hourly_station_data["start_station_name"].value_counts()

# Get the most and least frequent stations
most_frequent_stations = station_freq.head(100).index
least_frequent_stations = station_freq.tail(100).index

# Filter data for most and least frequent stations
most_freq_data = hourly_station_data[
    hourly_station_data["start_station_name"].isin(most_frequent_stations)
]
least_freq_data = hourly_station_data[
    hourly_station_data["start_station_name"].isin(least_frequent_stations)
]

# One-hot encode 'day_name' and 'Condition'
most_freq_encoded = pd.get_dummies(
    most_freq_data, columns=["day_name", "Condition"], drop_first=True
)
least_freq_encoded = pd.get_dummies(
    least_freq_data, columns=["day_name", "Condition"], drop_first=True
)


# Prepare features and target for both datasets
def prepare_data(data):
    X = data.drop("BikeUndocked", axis=1)  # Features
    y = data["BikeUndocked"]  # Target
    X = X.drop(columns=["started_at"])  # Drop timestamp column
    station_freq_map = X["start_station_name"].value_counts().to_dict()
    X["start_station_name"] = X["start_station_name"].map(
        station_freq_map
    )  # Frequency encoding
    X = X.fillna(0)  # Handle missing values
    return X, y


X_most, y_most = prepare_data(most_freq_encoded)
X_least, y_least = prepare_data(least_freq_encoded)

# Split data into training and testing sets
X_train_most, X_test_most, y_train_most, y_test_most = train_test_split(
    X_most, y_most, test_size=0.2, random_state=42
)
X_train_least, X_test_least, y_train_least, y_test_least = train_test_split(
    X_least, y_least, test_size=0.2, random_state=42
)

# Define a pipeline and grid search
pipe_lr = Pipeline([("scl", StandardScaler()), ("reg", LinearRegression())])
param_grid_lr = {"reg__fit_intercept": [True, False], "reg__positive": [True, False]}
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)


# Define function to train and log a model
def train_and_log_model(X_train, X_test, y_train, y_test, run_name):
    with mlflow.start_run(run_name=run_name):
        grid_lr.fit(X_train, y_train)

        # Log the best parameters
        best_params = grid_lr.best_params_
        mlflow.log_params(best_params)

        # Evaluate on test data and log metrics
        y_pred = grid_lr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared score
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)  # Log R-squared score

        # Create an input example
        input_example = X_train.iloc[0:1]  # Take the first row as an example

        # Log the best model with the input example
        mlflow.sklearn.log_model(
            grid_lr.best_estimator_,
            "linear_regression_model",
            input_example=input_example,
        )

        print(f"{run_name} - Best parameters: {best_params}")
        print(f"{run_name} - Mean Squared Error: {mse}")
        print(f"{run_name} - R-squared Score: {r2}")


# Train and log models for most and least frequent stations
train_and_log_model(
    X_train_most, X_test_most, y_train_most, y_test_most, "Most Frequent Stations"
)
train_and_log_model(
    X_train_least, X_test_least, y_train_least, y_test_least, "Least Frequent Stations"
)

print(most_frequent_stations)
print(least_frequent_stations)


# Resample and aggregate function based on `start_station_id`
def resample_and_aggregate_by_station(data, frequency="h"):  # Use 'h' instead of 'H'
    # Ensure `started_at` is a datetime column
    data["started_at"] = pd.to_datetime(data["started_at"])
    # Set the timestamp as index for resampling
    data = data.set_index("started_at")
    # Group by `start_station_id` and resample
    aggregated_data = (
        data.groupby("start_station_name")
        .resample(frequency)
        .agg(
            {
                "month": "first",  # Take the first value for categorical columns
                "hour": "first",  # Same as above
                "day_name": "first",  # Same as above
                "duration": "sum",  # Sum durations for the period
                "distance_km": "sum",  # Sum distances for the period
                "Temperature (°F)": "mean",  # Average temperature
                "Humidity": "mean",  # Average humidity
                "Wind Speed": "mean",  # Average wind speed
                "Precip.": "sum",  # Total precipitation for the period
                "Condition": "first",  # Representative condition
                "BikeUndocked": "sum",  # Sum undocked bikes
            }
        )
        .reset_index()  # Reset index after resampling
    )
    return aggregated_data


# Debugging checks for missing columns
def debug_check_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"The following columns are missing in the DataFrame: {missing_columns}"
        )


# Validate columns before resampling
required_columns = [
    "start_station_name",
    "started_at",
    "month",
    "hour",
    "day_name",
    "duration",
    "distance_km",
    "Temperature (°F)",
    "Humidity",
    "Wind Speed",
    "Precip.",
    "Condition",
    "BikeUndocked",
]

debug_check_columns(most_freq_data, required_columns)
debug_check_columns(least_freq_data, required_columns)

# Resample the most and least frequent data by `start_station_id`
resampled_most_freq = resample_and_aggregate_by_station(
    most_freq_data, frequency="h"
)  # Hourly resampling
resampled_least_freq = resample_and_aggregate_by_station(
    least_freq_data, frequency="h"
)  # Hourly resampling

# One-hot encode `day_name` and `Condition`
resampled_most_encoded = pd.get_dummies(
    resampled_most_freq, columns=["day_name", "Condition"], drop_first=True
)
resampled_least_encoded = pd.get_dummies(
    resampled_least_freq, columns=["day_name", "Condition"], drop_first=True
)

# Prepare features and target for resampled datasets
X_resampled_most, y_resampled_most = prepare_data(resampled_most_encoded)
X_resampled_least, y_resampled_least = prepare_data(resampled_least_encoded)

# Split data into training and testing sets for resampled data
(
    X_train_resampled_most,
    X_test_resampled_most,
    y_train_resampled_most,
    y_test_resampled_most,
) = train_test_split(X_resampled_most, y_resampled_most, test_size=0.2, random_state=42)
(
    X_train_resampled_least,
    X_test_resampled_least,
    y_train_resampled_least,
    y_test_resampled_least,
) = train_test_split(
    X_resampled_least, y_resampled_least, test_size=0.2, random_state=42
)

# Train and log new models for resampled data
train_and_log_model(
    X_train_resampled_most,
    X_test_resampled_most,
    y_train_resampled_most,
    y_test_resampled_most,
    "Resampled Most Frequent Stations by Start Station ID",
)

train_and_log_model(
    X_train_resampled_least,
    X_test_resampled_least,
    y_train_resampled_least,
    y_test_resampled_least,
    "Resampled Least Frequent Stations by Start Station ID",
)
