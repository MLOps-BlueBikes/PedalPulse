import streamlit as st
from google.cloud import aiplatform

# Initialize Vertex AI
PROJECT_ID = "blue-bike-prediction"
REGION = "us-central1"

def get_latest_endpoint(project_id, region):
    """
    Fetch the most recently created endpoint from Vertex AI.
    """
    aiplatform.init(project=project_id, location=region)
    endpoints = aiplatform.Endpoint.list()
    if not endpoints:
        st.error("No endpoints found in Vertex AI.")
        return None
    # Sort by creation time (most recent first)
    latest_endpoint = sorted(endpoints, key=lambda e: e.create_time, reverse=True)[0]
    return latest_endpoint.resource_name

# Get the latest endpoint dynamically
st.title("Blue Bike Prediction")
endpoint_id = get_latest_endpoint(PROJECT_ID, REGION)

if endpoint_id:
    endpoint = aiplatform.Endpoint(endpoint_id)
    # Streamlit form to input the test data
    with st.form(key="input_form"):
        start_station_name = st.number_input('Start Station Name', min_value=0, max_value=100)
        selected_date = st.date_input("Select Date")
        selected_time = st.time_input("Select Time")

        # Extract month and hour from the inputs
        month = selected_date.month
        hour = selected_time.hour
        duration = st.number_input('Duration', min_value=0.0, max_value=1000.0)
        distance_km = st.number_input('Distance (km)', min_value=0.0, max_value=1000.0)
        temperature = st.number_input('Temperature (°F)', min_value=-50.0, max_value=150.0)
        humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
        wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=150.0)
        precip = st.number_input('Precipitation', min_value=0.0, max_value=100.0)

        day_name = selected_date.strftime('%A')
        condition = st.radio('Weather Condition', [
            'Cloudy / Windy', 'Fair', 'Fair / Windy', 'Fog', 
            'Fog / Windy', 'Haze', 'Heavy Rain', 
            'Heavy Rain / Windy', 'Heavy T-Storm', 'Heavy T-Storm / Windy', 
            'Light Drizzle', 'Light Drizzle / Windy', 'Light Freezing Rain', 
            'Light Rain', 'Light Rain / Fog', 'Light Rain /Windy', 
            'Light Rain with Thunder', 'Light Snow', 'Light Snow / Windy', 
            'Mist', 'Mist / Windy', 'Mostly Cloudy', 
            'Mostly Cloudy / Windy', 'Partly Cloudy', 
            'Partly Cloudy / Windy', 'Patches of Fog', 'Rain', 
            'Rain / Windy', 'Snow / Fog', 'T-Storm', 
            'T-Storm / Windy', 'Thunder', 'Thunder / Windy', 
            'Wintry Mix', 'Wintry Mix / Windy'
        ])

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        all_days = [
            'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'
        ]
        all_conditions = [
            'Cloudy / Windy', 'Fair', 'Fair / Windy', 'Fog', 
            'Fog / Windy', 'Haze', 'Heavy Rain', 
            'Heavy Rain / Windy', 'Heavy T-Storm', 'Heavy T-Storm / Windy', 
            'Light Drizzle', 'Light Drizzle / Windy', 'Light Freezing Rain', 
            'Light Rain', 'Light Rain / Fog', 'Light Rain /Windy', 
            'Light Rain with Thunder', 'Light Snow', 'Light Snow / Windy', 
            'Mist', 'Mist / Windy', 'Mostly Cloudy', 
            'Mostly Cloudy / Windy', 'Partly Cloudy', 
            'Partly Cloudy / Windy', 'Patches of Fog', 'Rain', 
            'Rain / Windy', 'Snow / Fog', 'T-Storm', 
            'T-Storm / Windy', 'Thunder', 'Thunder / Windy', 
            'Wintry Mix', 'Wintry Mix / Windy'
        ]

        days = {f"day_name_{day}": 1 if day_name == day else 0 for day in all_days}
        conditions = {f"Condition_{cond}": 1 if condition == cond else 0 for cond in all_conditions}

        # Combine all features into test_data
        test_data = [[
            start_station_name, month, hour, duration, distance_km, temperature, humidity, wind_speed, precip,
            *days.values(),
            *conditions.values()
        ]]

        # Make the prediction
        response = endpoint.predict(instances=test_data)
        
        if response.predictions:  # Check if there are predictions
            predicted_bikes = round(response.predictions[0])  # Extract and round the first prediction
            st.write(f"Predicted number of bikes: {predicted_bikes}")
        else:
            st.error("No prediction returned from the model.")
else:
    st.error("Failed to retrieve the latest endpoint. Check Vertex AI configuration.")
