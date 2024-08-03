import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
df=pd.read_csv("C:\\Users\\saipr\\Downloads\\fake_and_real_news.csv\\fake_and_real_news.csv")
X=df['Text']
y=df['label']
#Text Preprocessing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
def preprocessing_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
X=df['Text'].apply(preprocessing_text)
#Converting into vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Streamlit app
def main():
    st.title("Text Classification")


    user_input = st.text_area("Enter your News")

    if st.button("Predict"):

        preprocessed_input = preprocessing_text(user_input)

        text_tfidf = vectorizer.transform([preprocessed_input])

        prediction = model.predict(text_tfidf)
        st.write(f"Prediction : {prediction[0]}")
    else:
        print("Enter your News")
if __name__ == "__main__":
    main()

