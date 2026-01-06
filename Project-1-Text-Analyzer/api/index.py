from flask import Flask, request, jsonify
from textblob import TextBlob
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

nltk_data_path = os.path.join('/tmp', 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except (LookupError, AttributeError):
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, AttributeError):
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

setup_nltk()

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.blob = TextBlob(text)

    def get_word_count(self):
        return len(self.text.split())

    def get_sentiment(self):
        score = self.blob.sentiment.polarity
        if score > 0.1: return "Positive"
        elif score < -0.1: return "Negative"
        return "Neutral"

    def get_keywords(self):
        cleaned = re.sub(r'[^\w\s]', '', self.text.lower())
        words = cleaned.split()
        stops = set(stopwords.words('english'))
        filtered = [w for w in words if w not in stops and len(w) > 2]
        return [word for word, count in Counter(filtered).most_common(3)]

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        setup_nltk()
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        analyzer = TextAnalyzer(data['text'])
        return jsonify({
            "word_count": analyzer.get_word_count(),
            "sentiment": analyzer.get_sentiment(),
            "keywords": analyzer.get_keywords()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500