import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("","", string.punctuation))
    text = text.strip()
    return text

def stem_text(text):
    return stemmer.stem(text)