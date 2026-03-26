import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    pkgs = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for pkg in pkgs:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if pkg.startswith('punkt') else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg)

download_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)