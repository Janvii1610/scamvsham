import streamlit as st
import pickle

cv = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(
    page_title="AI Spam Detection System",
    layout="centered"
)



#Page title
st.title("AI Spam Detection System")

st.write(
    "A machine learning web application that classifies SMS messages as spam or legitimate."
)

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.write("""
This application uses:
- Natural Language Processing (NLP)
- CountVectorizer
- Naive Bayes Classification
- Streamlit
""")

# Input box
input_sms = st.text_area("Enter Message")

# Predict button
if st.button("Analyze Message"):

    vector_input = cv.transform([input_sms])

    prediction = model.predict(vector_input)

    probability = model.predict_proba(vector_input)

    spam_probability = probability[0][1] * 100
    ham_probability = probability[0][0] * 100

    if prediction[0] == "spam":
        st.error("Spam message detected.")
    else:
        st.success("Legitimate message detected.")

    st.write(f"Spam Probability: {spam_probability:.2f}%")
    st.write(f"Legitimate Probability: {ham_probability:.2f}%")

# Footer
st.markdown("---")

st.caption(
    "Built using Python, Scikit-learn, NLP, and Streamlit."
)