import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize NLTK stop words
stop_words = set(stopwords.words('english'))


def read_documents(doc_folder):
    """
    Reads all text files in the specified directory and returns a dictionary
    with document ids as keys and the content as values.
    """
    documents = {}
    for doc_file in os.listdir(doc_folder):
        if doc_file.endswith('.txt'):
            doc_path = os.path.join(doc_folder, doc_file)
            with open(doc_path, 'r', encoding='utf-8') as file:
                documents[doc_file] = file.read()
    return documents


def preprocess_text(text):
    """
    Preprocesses the text by tokenizing, converting to lowercase, removing punctuation,
    and filtering out stop words.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Filter out stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def preprocess_documents(documents):
    """
    Apply text preprocessing to each document in the dictionary.
    """
    preprocessed_docs = []
    for doc_id, content in enumerate(documents):
        preprocessed_docs.append(preprocess_text(content))
    return preprocessed_docs
