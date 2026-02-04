import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# TITLE AND HEADER

st.set_page_config(page_title="Sentiment Analyzer", page_icon=" ")

st.title("Flipkart Review Sentiment Analyzer")
st.write("### Find out if a review is positive or negative!")

# PREPROCESSING FUNCTIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# LOAD MODEL

@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()


# MAIN APP

# Text input
review = st.text_area(
    "Enter a product review:",
    height=150,
    placeholder="Example: This product is amazing! Great quality and fast delivery."
)

# Predict button
if st.button("Analyze Sentiment", type="primary"):
    if review.strip():
        # Preprocess
        processed = preprocess(review)
        
        # Convert to numbers
        review_tfidf = tfidf.transform([processed])
        
        # Predict
        prediction = model.predict(review_tfidf)[0]
        probability = model.predict_proba(review_tfidf)[0]
        confidence = max(probability) * 100
        
        # Show results
        if prediction == 1:
            st.success("## POSITIVE REVIEW")
            st.balloons()
        else:
            st.error("## NEGATIVE REVIEW")
        
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence/100)
        
    else:
        st.warning("Please enter a review first!")


# SAMPLE BUTTONS


st.write("---")
st.write("**Or try a sample:**")

col1, col2 = st.columns(2)

with col1:
    if st.button("Positive Sample"):
        st.info("Amazing product! Highly recommended!")

with col2:
    if st.button("Negative Sample"):
        st.info("Terrible quality. Don't buy!")

# FOOTER

st.write("---")
st.write("Built with for Flipkart Sentiment Analysis Project")