import streamlit as st
import pickle
import string
import nltk
import os

# Create a directory for NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the custom path to NLTK's data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data to the custom directory
try:
    nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Now we can import these
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("Email/SMS Spam Classifier")
st.markdown("Enter the message below to check if it's spam or not.")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Result: Spam")
        else:
            st.header("Result: Not Spam")
    else:
        st.warning("Please enter a message to predict.")