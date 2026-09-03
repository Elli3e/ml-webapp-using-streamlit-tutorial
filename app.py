from pickle import load
import streamlit as st
import numpy as np
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import pickle



st.title("Stock Prediction App")

model = load_model("lstm_model.keras")
with open("scale_min_max.sav", "rb") as file:
    scale = pickle.load(file)

def stock_prediction ():

    selected_date = st.date_input("Select a date:")
    start_date = selected_date - timedelta(days=90)
    end_date = selected_date + timedelta(days=1)       

    data = yf.download(
    "BABA",
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)
   
    st.write("Downloaded data:")
    st.write(data)

    if data.empty:
        st.error("Yahoo Finance returned no data for this date range.")
        return None
        
    close = data[["Close"]]
    st.write("Close data:")
    st.write(close)

    if len(close) < 60:
        st.error(
            f"Only {len(close)} trading days were downloaded. "
            "At least 60 are required."
        )
        return None

    scaled_data = scale.transform(close)
    scaled_data_LSTM_input =scaled_data[-60: ] 
    scaled_data_LSTM_input=np.array(scaled_data_LSTM_input)
    reshaped_LSTM = scaled_data_LSTM_input.reshape(1, 60, 1)

    prediction = model.predict(reshaped_LSTM, verbose=0)
    prediction_original = scale.inverse_transform(prediction)

    return prediction_original

prediction = stock_prediction()
if prediction is not None:
    predicted_price = float(prediction[0][0])

    st.success(f"Predicted price for the next closing day: ${predicted_price:.2f}")