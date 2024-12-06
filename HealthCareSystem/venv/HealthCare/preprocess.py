import nltk
nltk.data.path.append('./nltk_data')  # Ensure this path points to your nltk_data directory

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer


# Initialize tools
spell = SpellChecker()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    # Correct spelling, remove stopwords, and handle None values
    words = [
        lemmatizer.lemmatize(spell.correction(word))
        for word in words
        if word not in stop_words and spell.correction(word) is not None
    ]
    return " ".join(words)
