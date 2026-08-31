import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Create stemmer
porter_stemmer = PorterStemmer()


# Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    stemmed_content = [
        porter_stemmer.stem(word)
        for word in stemmed_content
        if word not in stopwords.words('english')
    ]

    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


# Page title
st.title("📰 Fake News Detection")

st.write("Enter a news article below to check whether it is Real or Fake.")


# Text box
news = st.text_area(
    "Enter News:",
    placeholder="Paste your news article here..."
)


# Button
if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news first.")

    else:
        # Preprocess news
        processed_news = stemming(news)

        # Convert to TF-IDF
        news_tfidf = vectorizer.transform([processed_news])

        # Prediction
        prediction = model.predict(news_tfidf)

        # Display result
        if prediction[0] == 0:
            st.success("✅ This news is Real")
        else:
            st.error("🚨 This news is Fake")