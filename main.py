# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re
import streamlit as st
import pickle


def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())


tfidf = pickle.load(open('/vectorizer.pkl', 'rb'))
model = pickle.load(open('model (2).pkl', 'rb'))

st.title("hate speech detection")

input_sms = st.text_input("Enter the message -")

if st.button("predict"):

    # preprocess
    transformed_sms = process_tweet(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)
    # display
    if result != 1:
        st.header('hate speech')
    else:
        st.header('no hate speech')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
