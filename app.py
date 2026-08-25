import streamlit as st
import tensorflow as tf
import yfinance as yf
from pickle import load
import numpy as np

st.title("Stock Price Prediction")

with open ("scale_min_max.sav", "rb") as f:
 scale = load(f) 

prediction_model = tf.keras.models.load_model("lstm_model.keras")

ticker = st.text_input("Enter stock ticker", "BABA")

if st.button("Get Stock Data"):

    data = yf.download(ticker, period="1y")

    st.write("Data shape:", data.shape)
    st.write("Data columns:", data.columns)
    st.write(data.tail())

    if data.empty:
        st.error("Yahoo Finance returned no data.")
        st.stop()

    close_data = data[["Close"]]

    st.write("Close shape:", close_data.shape)
    st.write(close_data.tail())

    if close_data.empty:
        st.error("Close data is empty.")
        st.stop()

    scaled_stock = scale.transform(close_data)

    x_input = scaled_stock[-60:, 0]
    x_input = np.array(x_input)
    x_input = x_input.reshape(1, 60, 1)

    pred = prediction_model.predict(x_input)

    prediction = scale.inverse_transform(pred)

    st.write("Prediction price:", prediction[0][0])