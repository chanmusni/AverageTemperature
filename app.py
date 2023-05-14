import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import matplotlib.pyplot as plt

# load the saved LSTM model
model = load_model('lstm_model.h5')

# function to make predictions using the loaded model
def predict_temperature(data, num_days=60):
    # slice the data to only include the last num_days
    data = data[-num_days:]
    # reshape the input data to match the expected input shape of the model
    data = data.reshape((1, data.shape[0], 1))
    # make the prediction
    prediction = model.predict(data)
    # return the predicted temperature value
    return prediction[0][0]

# set up the Streamlit app
st.set_page_config(page_title="Daily Temperature Predictor", page_icon="🌡️")
st.title("Daily Temperature Predictor")
st.write("This app predicts the next 30 days' average temperature in Szeged, Hungary based on historical data.")

# load the temperature data
data = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
data.set_index('dt', inplace=True)

# display the current temperature data
st.subheader("Current Temperature Data")
st.write(data.tail())

# get the latest temperature value
latest_temp = data['LandAverageTemperature'].iloc[-1]

# get the date of the latest temperature value
latest_date = data.index[-1]

# make predictions for the next 30 days based on the last 365 days of data
num_days = 365
predicted_temps = []
for i in range(30):
    next_date = latest_date + pd.Timedelta(days=1)
    predicted_temp = predict_temperature(np.array(data['LandAverageTemperature']), num_days=num_days)
    predicted_temps.append(predicted_temp)
    latest_date = next_date
    data.loc[next_date] = predicted_temp
    
# display the predicted temperature values for the next 30 days
st.subheader("Next 30 Days' Predicted Temperatures")
st.write(data.tail(30)['LandAverageTemperature'])

# plot the predicted temperatures using Plotly Express
fig = px.line(data.tail(30), y='LandAverageTemperature', title="Predicted Temperatures' Graph for the Next 30 Days")
st.plotly_chart(fig)

# plot the predicted temperature values
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data['LandAverageTemperature'], label='Actual')
ax.plot(data['LandAverageTemperature'].tail(30), label='Predicted')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (Celsius)')
ax.set_title('Actual vs Predicted Daily Average Temperature')
ax.legend()
st.pyplot(fig)

# get the date and temperature of the highest predicted temperature
highest_temp = max(predicted_temps)
highest_temp_index = predicted_temps.index(highest_temp)
highest_temp_date = data.index[-30:][highest_temp_index]

# get the date and temperature of the lowest predicted temperature
lowest_temp = min(predicted_temps)
lowest_temp_index = predicted_temps.index(lowest_temp)
lowest_temp_date = data.index[-30:][lowest_temp_index]

# display the highest predicted temperature and its date
st.subheader("Highest Predicted Temperature")
st.write(f"The highest predicted temperature is {highest_temp:.2f}°C and it will happen on {highest_temp_date.date()}.")

# display the lowest predicted temperature and its date
st.subheader("Lowest Predicted Temperature")
st.write(f"The lowest predicted temperature is {lowest_temp:.2f}°C and it will happen on {lowest_temp_date.date()}.")
