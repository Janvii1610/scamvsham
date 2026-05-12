import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Input and output
x = df['message']
y = df['label']

# Convert text into numbers
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Streamlit UI
st.title("Spam Message Detector")

input_sms = st.text_area("Enter Message")

if st.button("Predict"):

    data = cv.transform([input_sms])

    prediction = model.predict(data)

    if prediction[0] == "spam":
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Ham Message")