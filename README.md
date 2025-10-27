<p align="center">
  <img src="https://github.com/yourusername/hinglish-sentiment-analysis/assets/banner.png" width="800" />
</p>

<h1 align="center">💬 Hinglish Sentiment Analysis using Regex Normalization & Machine Learning</h1>

<p align="center">
  <a href="https://github.com/yourusername/hinglish-sentiment-analysis/stargazers"><img src="https://img.shields.io/github/stars/yourusername/hinglish-sentiment-analysis?style=social" /></a>
  <a href="https://github.com/yourusername/hinglish-sentiment-analysis/network/members"><img src="https://img.shields.io/github/forks/yourusername/hinglish-sentiment-analysis?style=social" /></a>
  <a href="https://your-live-demo-link"><img src="https://img.shields.io/badge/Demo-Live-green" /></a>
  <a href="https://flask.palletsprojects.com"><img src="https://img.shields.io/badge/Built%20With-Flask-blue" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn-yellow" /></a>
</p>

---

## 🧠 Overview
> **Hinglish Sentiment Analysis** is a Natural Language Processing project that classifies sentiment (Positive / Negative / Neutral) in *Hinglish* — a mix of Hindi and English written in Roman script.  
> It uses **Regex-based normalization** for text cleaning and **Machine Learning models** (Logistic Regression, Naive Bayes) for classification.  
> The system achieves **93.33% accuracy** using a custom-built dataset of over 600 Hinglish sentences and is deployed through a simple **Flask web app**.

---

<details>
<summary><b>📘 Abstract (click to expand)</b></summary>

This project explores the intersection of computational linguistics and cultural linguistics by developing a Hinglish Sentiment Analysis system. Hinglish—an amalgamation of Hindi and English—is widely used in India’s multilingual digital communication. The system employs regex-based normalization techniques to clean noisy Hinglish text and leverages machine learning algorithms to classify sentiment as positive, negative, or neutral.

A custom dataset comprising 600+ manually curated Hinglish sentences across multiple domains (movies, food, travel, education, and daily life) was developed to train and evaluate the model. Logistic Regression achieved an accuracy of **93.33%**, outperforming Naive Bayes (**63.33%**). This demonstrates that even a moderately sized, high-quality dataset with robust preprocessing can yield highly accurate classification for code-mixed languages.

</details>

---

<details>
<summary><b>🎯 Problem Statement (click to expand)</b></summary>

Code-mixing—the blending of multiple languages—is common in multilingual societies. In India, users frequently write in Hinglish (a mix of Hindi and English) using the Roman alphabet.  
Conventional NLP tools assume monolingual input, making this a challenge.

The objective is to **automatically classify Hinglish text sentiment** (positive, negative, neutral) by handling:
- Informal spelling variations  
- Slang and emoticons  
- Code-switching  
- Romanized Hindi  

Our system addresses these challenges via regex normalization and robust ML training.

</details>

---

## 🧩 Objectives
- ✅ Develop regex-based preprocessing for Hinglish text  
- ✅ Build ML classifiers (SVM, Logistic Regression, Random Forest) using TF–IDF features  
- ✅ Evaluate with metrics (Accuracy, Precision, Recall, F1)  
- ✅ Deploy via Flask web application for real-time sentiment detection  

---

## 🧾 Dataset Description

The **Hinglish Sentiment Dataset** was custom-built to reflect real-world Hinglish usage.  
It contains ~600 labeled sentences across multiple domains, each expressing a clear emotional tone.

| Domain | Example Sentences | Sentiment | Notes |
|--------|------------------|------------|--------|
| 🎬 Movies | “Yaar this movie was awesome!”, “Bahut boring tha yeh film.” | Positive / Negative | Common Hinglish opinions |
| 🍔 Food | “Khana mast tha!”, “Taste boooring tha yaar.” | Positive / Negative | Restaurant and food reviews |
| ✈️ Travel | “Goa trip was superb yaar!”, “Crowded aur dirty jagah thi.” | Positive / Negative | Travel experiences |
| 💻 Tech | “Phone mast chal raha hai!”, “Bakwaas quality ka product.” | Positive / Negative | Product reviews |
| 📚 Education | “Exam mast gaya!”, “Workload bohot zyada hai.” | Positive / Negative | Student and work remarks |
| 🗣️ Daily Chat | “Chill yaar, sab theek hai.” | Neutral | Casual conversations |

📊 **Total Samples:** ~600  
Positive: ~200 | Negative: ~200 | Neutral: ~200  

---

## 🧹 Data Preprocessing & Regex Normalization

The preprocessing pipeline cleans and standardizes Hinglish text using **regular expressions**:

```python
# Regex Normalization Example
import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|https?://\S+|[^a-z\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # collapse long repetitions
    return text.strip()

sample = "Yaar this movie was totally awwwesome!!!!"
print(normalize_text(sample))
# Output: yaar this movie was totally awesome
