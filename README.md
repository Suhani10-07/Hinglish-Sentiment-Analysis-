

<h1 align="center" style="font-weight:700; font-size:2.2em;">
  🔍 Hinglish Sentiment Analysis  
  <br>
  <sub style="font-size:0.65em;">Regex Normalization • Machine Learning • Flask Web App</sub>
</h1>


---

## 🧠 Abstract
This project explores the intersection of computational linguistics and cultural linguistics by developing a Hinglish Sentiment Analysis system. Hinglish—an amalgamation of Hindi and English—is widely used in India’s multilingual social and digital conversations. The system employs regex-based normalization techniques to clean noisy Hinglish text and leverages machine learning algorithms to classify sentiment as positive, negative, or neutral.

A custom dataset comprising over 600 manually curated Hinglish sentences across multiple domains (movies, food, travel, education, and daily life) was developed to train and evaluate the model. Logistic Regression achieved an accuracy of **93.33%**, outperforming Naive Bayes (**63.33%**). This study demonstrates that even a moderately sized, high-quality dataset paired with robust preprocessing can yield highly accurate sentiment classification for code-mixed languages.

---

## 🎯 Problem Statement
Code-mixing, the practice of blending multiple languages in a single utterance, is common in multilingual societies. In India, many social media users write in Hinglish (a mix of Hindi and English) using the Roman alphabet. This poses significant challenges for NLP systems, since conventional tools assume monolingual input. For example, Hindi words written in Roman script do not match standard English or Hindi word lists, and non-standard spellings are common.

The goal is to automatically classify the sentiment (positive/negative/neutral) of such Hinglish text from social media. This involves handling informal language, slang, emoticons, irregular spellings and code-switching, which complicates both preprocessing and modeling. Our system must address these challenges by normalizing the raw text and training robust models on code-mixed data.

---

## 🎯 Objectives
- Develop a robust preprocessing pipeline for Hinglish text using regex rules to clean noise (mentions, URLs, punctuation, etc.) and normalize informal expressions.  
- Design and train machine learning classifiers (e.g., SVM, Logistic Regression, Random Forest) using TF–IDF features to predict sentiment.  
- Evaluate performance with standard metrics (accuracy, precision, recall, F1) and analyze error patterns (via confusion matrix).  
- Deploy the model in a user-friendly web application using Flask, allowing users to enter text and receive sentiment labels.

---

## 🧾 Dataset Description
For this project, a **custom Hinglish Sentiment Dataset** was developed to better capture the diversity of real-world Hinglish usage beyond social media tweets. The dataset contains over 600 manually curated samples representing day-to-day conversational Hinglish. Each sentence expresses a clear emotional tone and is labeled as Positive, Negative, or Neutral.

Unlike datasets limited to a single domain such as movie or product reviews, this dataset includes multi-domain content such as entertainment, education, travel, technology, health, and casual chat. This diversity ensures that the model can generalize well across various real-life communication contexts.

The dataset was saved in CSV format (`hinglish_sentiment_dataset_enhanced.csv`) and was also expanded to include regex-normalized text for preprocessing validation.

---

### 📊 Dataset Composition

| Domain | Example Sentences | Sentiment Labels | Remarks |
|--------|------------------|------------------|----------|
| 🎬 Movies & Shows | “Yaar this movie was awesome!”, “Bahut boring tha yeh film.” | Positive / Negative | Common Hinglish entertainment opinions |
| 🍔 Food & Dining | “Khana mast tha!”, “Taste boooring tha yaar.” | Positive / Negative | Restaurant and street food reviews |
| ✈️ Travel & Places | “Goa trip was superb yaar!”, “Crowded aur dirty jagah thi.” | Positive / Negative | Travel and experience sharing |
| 💻 Products & Tech | “Phone mast chal raha hai!”, “Bakwaas quality ka product.” | Positive / Negative | Product purchase experiences |
| 📚 Education & Work | “Exam mast gaya!”, “Workload bohot zyada hai yaar.” | Positive / Negative | Student and professional remarks |
| 🗣️ Daily Conversations | “Chill yaar, sab theek hai.”, “Pakka timepass tha!” | Neutral / Positive | Casual, chat-style Hinglish |
| ❤️ Health & Lifestyle | “Doctor ne acha treatment diya.”, “Bahut stress ho gaya.” | Positive / Negative | Common health-related sentiments |

📈 **Total Samples:** ~600  
Positive: ~200 | Negative: ~200 | Neutral: ~200

---

## 🧹 Data Preprocessing and Regex Normalization
The raw Hinglish tweets are highly informal and noisy. We perform iterative cleaning and normalization using **regular expressions** as the primary tool.

Key steps include:
- Lowercasing text  
- Removing Twitter tokens (mentions, hashtags, URLs)  
- Removing digits, punctuation, and special symbols  
- Expanding slang (e.g., “plz” → “please”)  
- Collapsing repeated characters (`soooo` → `soo`)  
- Handling emoticons and laughter (`lol` → `<laugh>`)  

Example normalization:
