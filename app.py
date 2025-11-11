import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake news Detector")
st.write("Enter news article  below to check if it's fake or real.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        transformed_input = vectorizer.transform([news_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == 1:
            st.error("The news article is likely FAKE.")
        else:
            st.success("The news article is likely REAL.")
    else:
        st.warning("Please enter a news article to check.")