from pickle import load 
import streamlit as st
import numpy as np


st.title("YouTube Spam Detector")
st.write("Enter a YouTube comment below to check whether it is spam.")

model = load(open("support_vector_youtube_spam.sav","rb"))
vectorizer = load(open("vectorize_youtubr_spam.sav","rb" ))

class_dict= {"0":"not_spam","1":"spam"}           


def predict_spam(comment):

       comment_tfidf = vectorizer.transform([comment])
       pred_class= model.predict(comment_tfidf)
       pred_class = int(pred_class[0])
       return class_dict[str(pred_class)]


# User input comment
comment = st.text_area("Enter a YouTube comment:",
                        placeholder="Example: Check out my channel and subscribe!"
)

# Predict button
if st.button("Predict"): 
    if comment.strip() == "": 
          st.warning("Please enter a YouTube comment first.")
    else: 
        prediction = predict_spam(comment)
        
        if prediction == "Spam": 
         st.error(f"Prediction: {prediction} 🚨")
        else:
         st.success(f"Prediction: {prediction} ✅")