# Project 1: Smart Text Analyzer 📊

Part of the **"Day 0 to Hero to become an AI engineer"** roadmap. This project serves as a foundational step into Natural Language Processing (NLP) by analyzing raw text data.

---

## 📝 Overview

The Smart Text Analyzer is a Python-based CLI tool that processes user input to provide statistical data, emotional sentiment, and key themes. It demonstrates the ability to clean data using Regular Expressions and perform basic NLP tasks using industry-standard libraries.

---

## ✨ Features

- **Statistical Analysis:** Word count, Sentiment, Top Keywords
- **Sentiment Analysis:** Categorizes text as **Positive**, **Negative**, or **Neutral** with a decimal polarity score.
- **Keyword Extraction:** Filters out "Stopwords" (common words like `the`, `is`, `at`) to identify the top 3 most meaningful words.

---

## 🛠️ Built With

- **Python 3.13.2**
- **TextBlob:** For sentiment analysis logic.
- **NLTK (Natural Language Toolkit):** For English stopword filtering.
- **RegEx (re):** For text cleaning and punctuation removal.

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/priyankamorajkar/Day-0-to-hero-to-become-an-AI-engineer.git
   ```

2. **Navigate to the project folder:**
   ```bash
   cd Project-1-Text-Analyzer
   ```

3. **Install dependencies:**
   ```bash
   pip install textblob nltk
   ```

4. **Run the script:**
    ```bash
    python TextAnalyzer.ipynb
    ```
---

## 📖 Example Output

Enter text: This was a terrible experience. The service was extremely slow and the staff was very rude. I waited for an hour just to get a cold meal. I am very disappointed and I will never come back to this place again. It was a total waste of money.
Word Count: 48
Sentiment: Negative
Top Keywords: [('terrible', 1), ('experience', 1), ('service', 1)]