#the final
import pytesseract
from PIL import Image

# Set the path to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import os
import re
import json
import pytesseract
import nltk 
nltk.data.path.append(r'D:\\nltk_data') 
nltk.download('punkt_tab')
# Download required NLTK data files
nltk.download('stopwords',download_dir='D:\\nltk_data')
nltk.download('punkt',download_dir='D:\\nltk_data')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from pdf2image import convert_from_path
from nltk.tokenize import sent_tokenize
import requests
from io import BytesIO
import tempfile


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.documents = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        # Tokenization and lowercasing
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords and apply stemming
        return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        tokens = self.tokenize(text)
        for token in tokens:
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)

    def save_index(self, path):
        with open(path, 'w') as f:
            json.dump({'index': self.index, 'documents': self.documents}, f)

    def load_index(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.index = defaultdict(list, data['index'])
            self.documents = data['documents']

    def search(self, query, num_results=5):
        tokens = self.tokenize(query)
        query_text = ' '.join(tokens)
        relevant_docs = set()
        for token in tokens:
            if token in self.index:
                relevant_docs.update(self.index[token])
        if not relevant_docs:
            return []
        # Calculate TF-IDF scores for relevant documents
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([self.documents[doc_id] for doc_id in relevant_docs])
        # Calculate cosine similarity between query and documents
        query_vector = tfidf_vectorizer.transform([query_text])
        cosine_similarities = tfidf_matrix.dot(query_vector.T).toarray().flatten()
        # Get indices of top matching documents
        top_indices = cosine_similarities.argsort()[-num_results:][::-1]
        return [list(relevant_docs)[i] for i in top_indices]

def read_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf_path = temp_pdf.name
    pdf_content = BytesIO(response.content)
    text = ''
    images = convert_from_path(temp_pdf_path)
    text=''
    for image in images:
        text += pytesseract.image_to_string(image)
    os.remove(temp_pdf_path)
    return text

def extract_relevant_sentences(document, query):
    sentences = sent_tokenize(document)
    query_lower = query.lower()
    relevant_sentences = [sentence for sentence in sentences if query_lower in sentence.lower()]
    return relevant_sentences

def query_index(inverted_index, query):
    relevant_document_ids = inverted_index.search(query)
    query_results = {}
    for doc_id in relevant_document_ids:
        relevant_sentences = extract_relevant_sentences(inverted_index.documents[doc_id], query)
        if relevant_sentences:
            query_results[doc_id] = relevant_sentences
    return query_results

def main():
    url = "https://vtu.ac.in/pdf/QP/BCHEM102set1.pdf"
    keyword = "applications"  # Change this to your keyword

    inverted_index = InvertedIndex()

    # Read and process the PDF from the URL
    text = read_pdf_from_url(url)
    inverted_index.add_document("document", text)

    # Save the index (optional)
    inverted_index.save_index("inverted_index.json")

    # Load the index (optional)
    inverted_index.load_index("inverted_index.json")

    # Query the index
    relevant_documents = query_index(inverted_index, keyword)

    print(f"Relevant sentences for the keyword '{keyword}':")
    for doc_id, sentences in relevant_documents.items():
        print("Document ID:", doc_id)
        for sentence in sentences:
            print(" -", sentence)
        print("-" * 80)

if __name__ == "__main__":
    main()
