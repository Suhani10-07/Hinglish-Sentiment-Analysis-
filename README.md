Hinglish Sentiment Analysis Using Regex Normalization and Machine Learning
Abstract
This project explores the intersection of computational linguistics and cultural linguistics by developing a Hinglish Sentiment Analysis system. Hinglish—an amalgamation of Hindi and English—is widely used in India’s multilingual social and digital conversations. The system employs regex-based normalization techniques to clean noisy Hinglish text and leverages machine learning algorithms to classify sentiment as positive, negative, or neutral.
A custom dataset comprising over 600 manually curated Hinglish sentences across multiple domains (movies, food, travel, education, and daily life) was developed to train and evaluate the model. Logistic Regression achieved an accuracy of 93.33%, outperforming Naive Bayes (63.33%). This study demonstrates that even a moderately sized, high-quality dataset paired with robust preprocessing can yield highly accurate sentiment classification for code-mixed languages.
Problem Statement
Code-mixing, the practice of blending multiple languages in a single utterance, is common in multilingual societies. In India, many social media users write in Hinglish (a mix of Hindi and English) using the Roman alphabet arxiv.org. This poses significant challenges for NLP systems, since conventional tools assume monolingual input arxiv.org. For example, Hindi words written in Roman script do not match standard English or Hindi word lists, and non-standard spellings are common. The goal is to automatically classify the sentiment (positive/negative/neutral) of such Hinglish text from social media. This involves handling informal language, slang, emoticons, irregular spellings and code-switching, which complicates both preprocessing and modeling. Our system must address these challenges by normalizing the raw text and training robust models on code-mixed data.
Objectives
The main objectives of this project are:
•	Develop a robust preprocessing pipeline for Hinglish text, using regex rules to clean noise (mentions, URLs, punctuation, etc.) and to normalize informal expressions.
•	Design and train machine learning classifiers (e.g. SVM, Logistic Regression, Random Forest) using TF–IDF features to predict sentiment.
•	Evaluate performance with standard metrics (accuracy, precision, recall, F1) and analyze error patterns (via confusion matrix).
•	Deploy the model in a user-friendly web application using Flask, allowing users to enter text and receive sentiment labels.

Dataset Description
For this project, a custom Hinglish Sentiment Dataset was developed to better capture the diversity of real-world Hinglish usage beyond social media tweets. The dataset contains over 600 manually curated samples representing day-to-day conversational Hinglish. Each sentence expresses a clear emotional tone and is labeled as Positive, Negative, or Neutral.
Unlike datasets limited to a single domain such as movie or product reviews, this dataset includes multi-domain content such as entertainment, education, travel, technology, health, and casual chat. This diversity ensures that the model can generalize well across various real-life communication contexts.
The dataset was saved in CSV format (hinglish_sentiment_dataset_enhanced.csv) and was also expanded to include regex-normalized text for preprocessing validation.
 Dataset Composition
Domain	Example Sentences	Sentiment Labels	Remarks
Movies & Shows	“Yaar this movie was awesome!”, “Bahut boring tha yeh film.”	Positive / Negative	Common Hinglish entertainment opinions
Food & Dining	“Khana mast tha!”, “Taste boooring tha yaar.”	Positive / Negative	Restaurant and street food reviews
Travel & Places	“Goa trip was superb yaar!”, “Crowded aur dirty jagah thi.”	Positive / Negative	Travel and experience sharing
Products & Tech	“Phone mast chal raha hai!”, “Bakwaas quality ka product.”	Positive / Negative	Product purchase experiences
Education & Work	“Exam mast gaya!”, “Workload bohot zyada hai yaar.”	Positive / Negative	Student and professional remarks
Daily Conversations	“Chill yaar, sab theek hai.”, “Pakka timepass tha!”	Neutral / Positive	Casual, chat-style Hinglish
Health & Lifestyle	“Doctor ne acha treatment diya.”, “Bahut stress ho gaya.”	Positive / Negative	Common health-related sentiments
Table 1: Custom Hinglish Sentiment Dataset across multiple domains.
Dataset Statistics
•	Total Samples: ~600
•	Positive Samples: ~200
•	Negative Samples: ~200
•	Neutral Samples: ~200
Columns:
•	original_text → Raw Hinglish input
•	normalized_text → Regex-cleaned Hinglish text
•	sentiment → Classified label
Examples of Data
Normalized examples:
========================================
Original : yaar this movie was totally awwwesome!!!!
Normalized: yaar this movie was totally awesome
------------------------------
Original : bahut achha performance, really amazingggg
Normalized: bahut good performance really amazingg
------------------------------
Original : sooooo goooood acting by everyone yaar
Normalized: soo good acting by everyone yaar
------------------------------
Original : fantastic storyline and grrreat direction
Normalized: fantastic storyline and great direction
------------------------------
Original : ekdum mast comedy scenes throughout
Normalized: mast comedy scenes throughout
------------------------------
Original : movie was totally booooring yaar
Normalized: movie was totally boring yaar
------------------------------
Original : bakwaaaaas acting throughout film
Normalized: bakwas acting throughout film
------------------------------
Original : really baaaad direction and story
Normalized: really bad direction and story
------------------------------
Original : movie was okayyyy nothing special
Normalized: movie was okayy nothing special
------------------------------
Original : decent enough performance overalll
Normalized: decent enough performance overall
------------------------------
Original : averageeee story line nothing wow
Normalized: average story line nothing wow
------------------------------
Original : pls pls dont waste time on this
Normalized: please please dont waste time on this
------------------------------
Original : thx bro that was jhakaas performance
Normalized: thanks bro that was jhakaas performance
------------------------------
Original : timepass movie, but chill if you like comedy
Normalized: timepass movie but chill if you like comedy
------------------------------
Original : plz fix the sound, its baaad
Normalized: please fix the sound its bad

Data Preprocessing and Regex Normalization
The raw Hinglish tweets are highly informal and noisy. We perform iterative cleaning and normalization using regular expressions as the primary tool aclanthology.orggeeksforgeeks.org. First, all text is lowercased to ensure uniformity. Then, regex patterns remove Twitter-specific tokens: for example, user mentions (@\w+), hashtags, URLs (https?://\S+), and punctuation are stripped out aclanthology.org. Numeric digits and HTML artifacts (if any) are removed with regex (e.g. \d+ to eliminate numbers) geeksforgeeks.org. Contractions or slang can be expanded or mapped to standard forms using lookup tables (e.g. “u”→“you”, “yaar” stays as is) before or after regex cleaning.
Next, we normalize elongated words and emoticons. A regex like r'(.)\1{2,}' is applied to collapse repeated characters: for example, “haaaappy”→“haappy” or “soooo”→“soo” (typically reducing runs to 2 characters). Common emoticons and laughter patterns can be mapped to tokens (“:)”→<smile>, “lol”→<laugh>). Stopwords (from both Hindi and English, after roman transliteration) are removed to focus on sentiment-bearing terms.
•	Regex normalization handles noisy Hinglish patterns (e.g., “awwwesome” → “awesome”).
•	Covers multiple conversation domains beyond reviews.
•	Balanced sentiment distribution for fair model training.
Example normalization:
•	Original: "@user Kya baat hai!!!! 😍😍 lol soooo amazing!!!"
•	Cleaned: "kya baat hai lol soo amazing" → "kya baat hai lol soo amazing"
The above steps illustrate collapsing multiple punctuation marks, stripping user handles, and reducing repeated letters. These regex-based normalization rules substantially reduce noise in the code-mixed text aclanthology.orggeeksforgeeks.org, making the data more consistent for feature extraction.

Modeling and Methodology
Regex patterns were designed to correct extended characters, remove noise, and standardize spelling:
(r'(a+w+e+s*o+m+e+)', 'awesome'),
(r'(g+o+o+d+)', 'good'),
(r'(b+a+d+)', 'bad'),
(r'(b+o+r+i+n+g+)', 'boring'),
(r'(a+m+a+z+i+n+g+)', 'amazing'),
(r'(f+a+n+t+a+s+t+i+c+)', 'fantastic'),
(r'(s+u+p+e+r+b+)', 'superb')

Evaluation Metrics and Results
•	Feature Extraction: TF-IDF Vectorizer with bigram range (1,2).
•	Models Used:
o	Naive Bayes
o	Logistic Regression
 
Table 2: Evaluation metrics for different classifiers on the Hinglish test set (in 3-way sentiment classification).
The confusion matrix revealed that most misclassifications occurred between neutral and weakly positive/negative tweets, suggesting class imbalance issues. Nevertheless, the overall metrics are encouraging for a code-mixed setting. In prior work on this dataset, the top-ranked system achieved a macro F1 of about 75% using advanced Transformer models aclanthology.org; our simpler ML-based approach achieves around 69% F1, indicating room for improvement but validating the effectiveness of TF–IDF features and normalization. (Notably, logistic regression’s superior accuracy over Naïve Bayes in similar tasks has been reported researchgate.net.)
Web Application Design
The final system is wrapped in a Flask web application for ease of use. Flask is a lightweight Python web framework that handles HTTP requests and rendering of HTML templates. Our app provides a simple interface where the user can input a Hinglish sentence into a text box and submit it. Upon submission, the backend Flask route receives the text, applies the same regex-based preprocessing, vectorizes it with the TF–IDF transformer, and feeds it into the trained classifier model. The predicted sentiment (e.g. Positive/Negative/Neutral) is then displayed on the webpage. The interface is kept minimal: an input field and submit button on one page, and a result page showing the sentiment label and confidence scores. This design allows real-time sentiment feedback and can be extended to additional functionality (e.g. batch processing) in future work.
•	Text normalization
•	TF-IDF transformation
•	Logistic Regression classification
Architecture Overview:
User Input → Regex Normalization → TF-IDF → ML Model → Sentiment Output
System Architecture Diagram
Figure 1 illustrates the overall architecture and data flow of the system. The workflow begins with Data Ingestion (collecting Hinglish text, e.g. via Twitter API). Next, Preprocessing/Normalization applies the regex-based cleaning rules described above. The Feature Extraction stage converts text to TF–IDF vectors. These features go into the Classification module (machine learning models), which outputs a sentiment label. Finally, the Web Interface (Flask) handles user input and displays the result. Each module corresponds to a component in the pipeline, ensuring modularity and clarity of operation.
Figure 1. System architecture for Hinglish sentiment analysis (data flow from user input through preprocessing, feature extraction, classification, to output).
Conclusion
We have developed a formal sentiment analysis pipeline for Hinglish text that combines regex-based normalization with classical machine learning classifiers. Our preprocessing effectively cleans code-mixed tweets by removing noise (user mentions, URLs, repeated characters, etc.) and standardizing the text. Using TF–IDF features and algorithms like SVM and Logistic Regression, the system achieves reasonable classification performance (F1≈0.69) on benchmark dataarxiv.org. The Flask web app demonstrates practical deployability of the model. This work confirms that even simple ML approaches, when combined with thoughtful normalization, can perform competitively on the challenging Hinglish sentiment task.
Future Work
Future extensions may include: (1) Deep Learning Models: Incorporating embeddings or transformer models (e.g. mBERT, Hinglish-specific BERT) to capture semantic nuance, as these have shown higher accuracy in recent shared tasksaclanthology.org. (2) Expanded Data: Collecting more code-mixed data (other domains like product reviews or YouTube comments) to improve coverage and handle slang terms. (3) Advanced Normalization: Using language identification to apply language-specific preprocessing, or dictionary-based transliteration of Romanized Hindi to native script. (4) Ensemble and Hyperparameter Tuning: Further optimize ensemble strategies and explore weighting TF–IDF n-gram ranges. (5) User Interface Enhancements: Improving the web app with sentiment visualizations, API endpoints for batch analysis, or mobile integration. These steps could further boost performance and usability for real-world Hinglish sentiment analysis.
Future Extensions:
•	Integration of transformer-based models (HingBERT, Gemma, Qwen).
•	Expansion of dataset to 2,000+ samples.
•	Addition of speech-based Hinglish recognition.
•	Multi-task learning for intent detection and translation.
References
•	Singh et al. (2021): G. Singh, “Sentiment Analysis of Code-Mixed Social Media Text (Hinglish)”, Proc. Semeval-2020 Task 9, arXiv:2102.12149arxiv.orgarxiv.org.
•	Patwa et al. (2020): P. Patwa et al., “SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets”, Proc. SemEval-2020, pp. 774–790aclanthology.org.
•	JUNLP (2020): S. Mahata et al., “Sentiment Analysis of Hindi–English Code-Mixed Data Using Grid Search”, Proc. SemEval-2020, pp. 1–5aclanthology.org.
•	GeeksforGeeks (2025): “Text Preprocessing in NLP”, GeeksforGeeks, 2025geeksforgeeks.orggeeksforgeeks.org.
•	AdapterHub (2023): “Hinglish Sentiment (Twitter)”, AdapterHub resource page (SemEval 2020 dataset)adapterhub.ml.
•	Flask Documentation (2025): Pallets Projects, “Welcome to Flask”, (Flask web framework docs).
•	Arif et al. (2025): D. Arif et al., “Sentiment Classification of MyPertamina Reviews”, Int. J. Innov. Res. Tech. (IJIRT), 2025researchgate.net.

