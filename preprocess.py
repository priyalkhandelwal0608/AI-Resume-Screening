import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure all required NLTK resources are available
nltk.download('punkt')
nltk.download('punkt_tab')   # Fix for LookupError
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and normalize text for NLP tasks:
    - Lowercase
    - Remove digits
    - Remove punctuation
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = nltk.word_tokenize(text)  # split into words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)