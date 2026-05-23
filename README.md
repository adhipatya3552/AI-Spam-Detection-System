# 📩 AI Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Preprocessing-green?style=for-the-badge)
![Naive Bayes](https://img.shields.io/badge/Naive%20Bayes-Classifier-blueviolet?style=for-the-badge)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Classifier-yellow?style=for-the-badge)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-spam-detection-system-adhipatya.streamlit.app/)

**A full AI-powered Spam and Threat Detection system that uses NLP, TF-IDF, Naive Bayes, and Logistic Regression to detect spam messages and phishing emails — with heuristic-based threat detection, real mailbox parsing, automatic model selection, and both a CLI and a Streamlit web interface.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How the System Works](#-how-the-system-works)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Running the Project](#-running-the-project)
- [Training the Model](#-training-the-model)
- [Evaluating the Model](#-evaluating-the-model)
- [Word Cloud Visualization](#-word-cloud-visualization)
- [Mbox Parser and Anonymizer](#-mbox-parser-and-anonymizer)
- [Module Reference](#-module-reference)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)

---

## 🧠 Overview

This project is an **AI-powered Spam and Threat Detection System** built entirely in Python. It can analyze any text message or email and decide whether it is **spam** or **safe (ham)**, along with a confidence percentage.

What makes this system different from a basic spam classifier is that it combines two layers of detection:

- 🛡️ **Heuristic Detection** — a rule-based layer that catches high-confidence phishing patterns, urgency tactics, and Bayesian poisoning tricks before the ML model even runs
- 🤖 **Machine Learning Detection** — a TF-IDF + ML pipeline (Naive Bayes and Logistic Regression are both trained, and the better one is automatically saved) that classifies any message it didn't already catch through heuristics

On top of that, the project includes a **custom real email parser** that can handle raw MIME/SMTP email format, extract meaningful text from HTML emails, strip hidden invisible text used in Bayesian poisoning attacks, and parse real `.mbox` mailbox files to build a custom training dataset — all while anonymizing personal information.

You can run it as a **menu-driven CLI** using `main.py`, or as a fully interactive **Streamlit web app** using `app/app.py`.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Dual ML Models** | Both Naive Bayes and Logistic Regression are trained; the better-performing one is automatically saved |
| 🛡️ **Heuristic Detection** | Rule-based layer checks for 15+ phishing phrases and urgency patterns before the ML model runs |
| 🕵️ **Bayesian Poisoning Detection** | Detects hidden invisible text (`display:none`) injected into HTML emails to fool spam filters |
| 📧 **Full Email Parser** | Parses raw MIME/SMTP email content, extracts subject, body, and sender from both plain text and HTML emails |
| 📂 **Mbox Mailbox Parser** | Reads a real `.mbox` file, extracts all emails, anonymizes personal info, and saves them as a custom CSV dataset |
| 🧹 **Text Preprocessing** | Lowercases, removes special characters, and strips NLTK stopwords before classification |
| 📊 **Model Evaluation** | Prints accuracy, precision, recall, F1 score, and a confusion matrix heatmap |
| ☁️ **Word Cloud Visualization** | Generates separate word clouds for spam and ham messages to visually compare their vocabulary |
| ⚡ **Live Message Scanning** | Paste any message in the web app and get an instant spam prediction with confidence |
| 📂 **Email File Upload** | Upload a `.txt` email file, preview it cleaned or raw, and scan it for spam |
| 🎯 **Risk Level Classification** | Confidence scores are mapped to HIGH (>85%), MEDIUM (>60%), or LOW risk levels |
| 🎨 **Custom Dark Theme** | Streamlit app uses a custom dark red theme configured via `.streamlit/config.toml` |
| 🖥️ **CLI Interface** | Menu-driven terminal app with options to train, evaluate, and predict |
| 💾 **Saved Model Pipeline** | Full sklearn Pipeline (TF-IDF + model) saved as `.pkl` using joblib for fast reuse |

---

## ⚙️ How the System Works

The project is broken into well-separated modules, each handling one specific step. Here is a complete walkthrough of every stage:

---

### Step 1 — Data Loading (`src/data_loader.py`)

The raw dataset (`data/spam.csv`) is loaded using Pandas. The CSV has two columns: `label` (which contains either `"ham"` or `"spam"`) and `text` (the message content).

The `label` column is immediately mapped to numbers — `ham` becomes `0` and `spam` becomes `1` — so the ML model can work with it numerically.

If a custom anonymized dataset (`data/anonymized_spam.csv`) is also present, it gets merged with the main dataset during training. This allows the model to learn from real personal spam emails collected from an actual mailbox — giving it better real-world coverage than the base dataset alone.

> **In simple words:** It loads and combines the training data and converts text labels into numbers the model understands.

---

### Step 2 — Text Preprocessing (`src/preprocess.py`)

Raw message text is messy and inconsistent. Before any ML model can work with it, the text goes through three cleaning steps:

1. **Lowercase conversion** — "FREE" and "free" become the same word
2. **Special character removal** — everything that is not a letter, number, or space is stripped out using a regex pattern `[^a-zA-Z0-9 ]`
3. **Stopword removal** — common English words like "the", "is", "your", "and" that appear in almost every message and carry no spam signal are removed using NLTK's stopwords list

The same `clean_text()` function is applied to both training data and any new message before prediction — this ensures consistency.

> **In simple words:** It strips all the noise from the text and keeps only the words that actually carry meaning for spam detection.

---

### Step 3 — Model Training (`src/train_model.py`)

This is where the actual machine learning happens. Both datasets are loaded, cleaned, and then two models are trained and compared:

**TF-IDF Vectorization:**
Before feeding text to any ML model, it must be converted to numbers. `TfidfVectorizer` does this by calculating how important each word is in a message relative to all messages in the dataset. Common words that appear in every message get lower weights; rare but distinctive words (like "FREE", "WIN", "CLAIM") that appear mostly in spam get higher weights.

**Both models are wrapped in a scikit-learn `Pipeline`**, which chains TF-IDF vectorization and the classifier together as one object. This is important because it means during prediction, any new message only needs to be passed to the pipeline once — vectorization and classification happen in one shot.

**Model 1 — Naive Bayes (`MultinomialNB`):**
Naive Bayes is a probabilistic classifier that is especially fast and effective for text classification. It calculates the probability that a message belongs to each class (spam or ham) based on which words appear in it. It works very well on short messages like SMS.

**Model 2 — Logistic Regression (`LogisticRegression`):**
Logistic Regression is a linear classifier that finds the best boundary between spam and ham in the TF-IDF feature space. It tends to perform slightly better on larger and more varied datasets because it can learn more complex patterns.

**Automatic Best Model Selection:**
After both models are trained and their accuracy scores are compared on the same test set, the one with the higher accuracy is automatically saved as `models/spam_pipeline.pkl` using `joblib`. This makes the system self-improving — if you add more data and retrain, it will always save whichever approach works best for that particular dataset.

> **In simple words:** Two ML models are trained and compared head to head on a test split, and whichever wins gets saved as the final model.

---

### Step 4 — Prediction with Heuristic Override (`src/predict.py`)

This is the most advanced and complex module in the project. When a new message comes in for prediction, it goes through multiple layers before the ML model is even consulted.

#### Layer 1 — Email Parsing

If the input looks like a raw email (with MIME headers like `From:`, `Subject:`, `Content-Type:`), Python's built-in `email` library parses it properly:

- For **multipart emails** (emails with both plain text and HTML versions), it extracts the plain text part first
- If no plain text is available, it extracts the **HTML part** and passes it to the custom `EmailHTMLStripper`
- The `Subject:` and `From:` fields are also extracted and included in the text that gets analyzed

If the input is just a plain message (not a MIME email), it is used as-is.

#### Layer 2 — HTML Stripping and Bayesian Poisoning Detection

The custom `EmailHTMLStripper` class (which extends Python's built-in `HTMLParser`) does two things:

**Visible content extraction:** It reads through the HTML and collects only the text that would actually appear on screen — ignoring tags like `<style>`, `<script>`, and `<head>`.

**Hidden text detection (Bayesian Poisoning):** Some spam emails inject large amounts of invisible text inside elements styled with `display:none`. This hidden text is filled with legitimate-looking words specifically to confuse ML-based spam filters — a technique called **Bayesian poisoning**. The `EmailHTMLStripper` tracks text inside hidden elements separately and measures how much of it there is. If there is more than 150 characters of hidden text, the message is immediately flagged as spam with 99.5% confidence — before any ML classification.

#### Layer 3 — Heuristic Phrase Matching

Even before the ML model runs, the subject line and body are scanned for **15 high-confidence phishing and urgency phrases**:

```
"blocked your account", "renew your subscription",
"photos and videos will be deleted", "account will be deleted",
"storage has reached", "sync has been paused",
"unable to renew", "permanently removed",
"final notice", "upgrade storage",
"unauthorized access", "verify your identity",
"verify your account", "billing information", "critical limit"
```

Three heuristic rules are checked in order:
1. If **hidden text length > 150** characters → spam at 99.5% confidence (Bayesian poisoning)
2. If **2 or more phishing phrases** match in subject or body → spam at 97.5% confidence
3. If **urgent subject line phrases** match → spam at 95.0% confidence

If any of these rules fire, the result is returned immediately and the ML model is skipped entirely. This makes the system much faster and more reliable for well-known phishing patterns.

#### Layer 4 — ML Model Classification

If the heuristics don't catch anything, the cleaned message text is passed to the saved `spam_pipeline.pkl` for classification. The pipeline:
1. Vectorizes the cleaned text using TF-IDF
2. Predicts the class (0 = ham, 1 = spam)
3. Returns the **probability score** via `predict_proba()`, which is multiplied by 100 to give the final confidence percentage

> **In simple words:** The system first checks for known phishing tricks using rules, and only if nothing matches does it run the ML model. This layered approach makes it more reliable than ML alone.

---

### Step 5 — Model Evaluation (`src/evaluate_model.py`)

After training, the model can be evaluated on the test split of the dataset. It prints four standard metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Percentage of all messages (both spam and ham) classified correctly |
| **Precision** | Of all messages flagged as spam, how many were actually spam |
| **Recall** | Of all actual spam messages, how many did the model catch |
| **F1 Score** | Harmonic mean of Precision and Recall — balances both |

It also prints a full `classification_report` and displays a **Confusion Matrix heatmap** using Seaborn, showing exactly how many true positives, false positives, true negatives, and false negatives occurred.

> **In simple words:** It tells you how well the model is actually performing, not just what its accuracy number is.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        USER INPUT                              │
│         (Text Message / Email / Uploaded .txt File)            │
└───────────────────────────────┬───────────────────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │      app/app.py          │
                   │   (Streamlit Web UI)     │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │      predict.py          │
                   │  (Multi-layer Detection) │
                   └──┬──────────────────┬───┘
                      │                  │
         ┌────────────▼──┐    ┌──────────▼──────────┐
         │  Email Parser  │    │  Heuristic Checker   │
         │ (MIME/SMTP →   │    │  (15 phishing rules  │
         │  plain text)   │    │  + Bayesian poison   │
         └────────────┬──┘    │   detection)          │
                      │       └──────────┬────────────┘
         ┌────────────▼──────────────────▼────────────┐
         │         EmailHTMLStripper                    │
         │  (Strip HTML, detect display:none hidden     │
         │   text, measure Bayesian poisoning length)   │
         └────────────────────────┬────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │    preprocess.py         │
                     │ (lowercase → regex clean │
                     │  → remove stopwords)     │
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   spam_pipeline.pkl      │
                     │  (TF-IDF Vectorizer +    │
                     │   Best ML Classifier)    │
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   Prediction Result      │
                     │  (Spam/Ham + Confidence  │
                     │   % + Risk Level)        │
                     └─────────────────────────┘
```

**Full Training Data Flow:**

```
spam.csv  +  anonymized_spam.csv (from Spam.mbox)
                        │
                        ▼
               data_loader.py  ──▶  DataFrame (label: 0/1, text)
                        │
                        ▼
               preprocess.py  ──▶  Cleaned text
                        │
                        ▼
               train_model.py
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
       Naive Bayes           Logistic Regression
       Pipeline              Pipeline
       (TF-IDF + NB)         (TF-IDF + LR)
              │                    │
              └─────────┬──────────┘
                        │
              Compare accuracy on test set
                        │
                        ▼
              Save best model as
              models/spam_pipeline.pkl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Spam.mbox  ──▶  parse_mbox.py  ──▶  anonymized_spam.csv
(real emails)   (extract + anonymize   (custom training data)
                 emails, URLs, phones,
                 names)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **Web UI** | Streamlit | Browser-based detection interface |
| **ML Models** | Scikit-Learn `MultinomialNB`, `LogisticRegression` | Spam classification |
| **Vectorization** | Scikit-Learn `TfidfVectorizer` | Converting text to TF-IDF numerical vectors |
| **ML Pipeline** | Scikit-Learn `Pipeline` | Chains TF-IDF + model into one reusable object |
| **Text Preprocessing** | NLTK | Stopword removal and text normalization |
| **Email Parsing** | Python `email` (stdlib) | Parsing raw MIME/SMTP email content |
| **HTML Stripping** | Python `html.parser` (stdlib) | Stripping HTML, detecting hidden Bayesian poisoning text |
| **Mailbox Parsing** | Python `mailbox` (stdlib) | Reading `.mbox` files |
| **Model Saving** | Joblib | Saving and loading the trained pipeline from disk |
| **Data Handling** | Pandas | Loading and merging CSV datasets |
| **Visualization** | Matplotlib, Seaborn | Confusion matrix heatmap |
| **Word Cloud** | WordCloud | Spam vs ham word cloud generation |
| **Theme** | `.streamlit/config.toml` | Custom dark red Streamlit theme |

---

## 📁 Project Structure

```
spam-detection-system/
│
├── app/
│   └── app.py                      # Streamlit web app (live scan, file upload, risk levels)
│
├── data/
│   ├── spam.csv                    # Base SMS spam dataset (label, text)
│   ├── anonymized_spam.csv         # Custom spam dataset extracted from Spam.mbox
│   ├── Spam.mbox                   # Real personal spam mailbox file (input for parse_mbox.py)
│   └── *.txt                       # User-provided email files for testing (user must download/provide their own mail .txt)
│
├── models/
│   └── spam_pipeline.pkl           # Saved best model (TF-IDF + NB or LR) — auto-generated
│
├── notebooks/
│   └── eda.ipynb                   # Jupyter notebook for dataset exploration
│
├── src/
│   ├── data_loader.py              # Loads spam.csv, maps ham→0 and spam→1
│   ├── preprocess.py               # Cleans text (lowercase, regex, NLTK stopwords)
│   ├── train_model.py              # Trains NB and LR, compares accuracy, saves best model
│   ├── evaluate_model.py           # Evaluates model: accuracy, precision, recall, F1, confusion matrix
│   ├── predict.py                  # Full prediction pipeline: email parsing, heuristics, ML
│   ├── parse_mbox.py               # Parses Spam.mbox, anonymizes emails, saves anonymized_spam.csv
│   ├── wordcloud_visualization.py  # Generates spam and ham word cloud images
│   └── utils.py                    # Helper functions (create_directory, print_separator)
│
├── .streamlit/
│   └── config.toml                 # Custom dark theme config for the Streamlit app
│
├── main.py                         # CLI entry point (Train / Evaluate / Predict menu)
└── requirements.txt                # Python dependencies
```

---

## 📊 Dataset

### 1. Base SMS Spam Dataset (`data/spam.csv`)

This is the well-known **SMS Spam Collection Dataset** — a public dataset widely used for spam classification research. The dataset file 'spam.csv' was taken from Kaggle as its source - "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset"

| Column | Description |
|--------|-------------|
| `label` | Message category — `"ham"` (safe) or `"spam"` |
| `text` | Raw message content |

After loading, `label` is mapped to: `ham → 0`, `spam → 1`.

### 2. Custom Anonymized Spam Dataset (`data/anonymized_spam.csv`)

For users to check spam detection accuracy, they can download their own `.mbox` file which contains their own Spam emails from online services such as Gmail, Yahoo, etc. by looking online and provide that file to the `parse_mbox.py` script to train the model, which will create the file `data/anonymized_spam.csv` and will use it for training the model along with the base dataset.

This is a **custom dataset built from a real personal spam mailbox** (`data/Spam.mbox`). The `parse_mbox.py` script reads all emails from the mailbox, extracts their text content, and scrubs all personal information before saving them as training data.

The anonymization replaces:
- **Email addresses** → `[email]`
- **URLs** → `[url]`
- **Phone numbers** → `[phone]`
- **Personal names** → `[name]`

This dataset gives the model exposure to modern, real-world phishing and spam emails that may not be covered by the SMS-only base dataset.

### 3. Sample Phishing Email Files (`data/*.txt`)

Users will have to download or provide their own email `.txt` files for testing the upload feature in the Streamlit app.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above
- pip

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/spam-detection-system.git
cd spam-detection-system
```

---

### 2. Create a Virtual Environment

It is recommended to create a virtual environment first so all libraries get installed in an isolated space and don't interfere with your system Python.

```bash
# Create virtual environment
python -m venv venv

# Activate — on Windows
venv\Scripts\activate

# Activate — on Mac/Linux
source venv/bin/activate
```

---

### 3. Install Dependencies

Once the virtual environment is active:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
nltk
wordcloud
```

---

### 4. Download NLTK Stopwords

The preprocessing module uses NLTK stopwords. This downloads automatically when the code runs, but you can also download it manually once:

```python
import nltk
nltk.download("stopwords")
```

---

## ▶️ Running the Project

### Option 1 — Streamlit Web App (Recommended)

You can access the live deployed application online here:
👉 **[AI Spam Detection System App](https://ai-spam-detection-system-adhipatya.streamlit.app/)**

Alternatively, to run the Streamlit app locally:
```bash
streamlit run app/app.py
```

The local app will open in your browser at `http://localhost:8501`.

**What you can do in the app:**

**📊 Dataset Distribution** — A bar chart at the top shows the class balance between spam and ham messages in the training dataset.

**⚡ Live Message Scanning:**
1. Type or paste any message or email content into the text area
2. The result appears instantly — no button needed
3. You will see:
   - ✅ **Safe Message** (green) or 🚨 **Spam Message Detected** (red)
   - A **confidence percentage** and a **progress bar**
   - A **risk level** badge: HIGH RISK (>85%), MEDIUM RISK (>60%), or LOW RISK

**📂 Email File Upload:**
1. Click **Upload a .txt email file** and select any `.txt` email file
2. If the file contains email headers (MIME format), a toggle appears to switch between **Cleaned Preview** (extracted body) and **Raw Email Code** (original content)
3. Click **Scan Uploaded File** to analyze it

---

### Option 2 — CLI (Terminal)

```bash
python main.py
```

You will see this menu:

```
===== AI SPAM DETECTION SYSTEM =====

1. Train Model
2. Evaluate Model
3. Predict Message
4. Exit

Enter your choice:
```

- **Option 1** — Trains both Naive Bayes and Logistic Regression, compares accuracy, saves the winner
- **Option 2** — Evaluates the saved model and shows accuracy, precision, recall, F1, and confusion matrix
- **Option 3** — Prompts you to type a message and returns spam/ham with confidence percentage

---

## 🧠 Training the Model

```bash
python src/train_model.py
```

Or from the CLI menu, select option **1**.

**What happens:**
1. `spam.csv` is loaded and labels are mapped to 0/1
2. If `anonymized_spam.csv` exists, it is merged in to expand the training data
3. All text is cleaned using `preprocess.py`
4. Data is split 80/20 (train/test) with `random_state=42` for reproducibility
5. Both models are trained and tested:

```
Naive Bayes Accuracy:         0.9820
Logistic Regression Accuracy: 0.9874

Best Model: Logistic Regression
Model saved successfully.
```

6. The better model's pipeline is saved to `models/spam_pipeline.pkl`

> ⚠️ **Training is required before running any prediction.** The model file is already included in the repository, but if you add new data or want to retrain, run this step.

---

## 📈 Evaluating the Model

```bash
python src/evaluate_model.py
```

Or from the CLI menu, select option **2**.

**Sample output:**

```
===== MODEL EVALUATION =====

Accuracy  : 0.9874
Precision : 0.9762
Recall    : 0.9651
F1 Score  : 0.9706

===== CLASSIFICATION REPORT =====

              precision    recall  f1-score   support

           0       0.99      0.99      0.99       966
           1       0.98      0.97      0.97       149

    accuracy                           0.99      1115
```

A **Confusion Matrix heatmap** is also displayed showing true positives, false positives, true negatives, and false negatives broken down visually.

---

## ☁️ Word Cloud Visualization

```bash
python src/wordcloud_visualization.py
```

This generates two word cloud images:

- **Spam Word Cloud** (black background) — shows the most frequently appearing words across all spam messages. Expect to see words like "free", "call", "claim", "win", "prize"
- **Ham Word Cloud** (white background) — shows the most frequent words in safe messages. These tend to be everyday conversational words

This is useful for understanding what vocabulary the model has learned to associate with spam vs legitimate messages.

---

## 📬 Mbox Parser and Anonymizer

If you want to add your own real spam emails to the training data:

1. Export your spam folder from Gmail or any email client as a `.mbox` file
2. Place it at `data/Spam.mbox`
3. Run:

```bash
python src/parse_mbox.py
```

**What it does:**
- Reads every email in the mailbox using Python's `mailbox` library
- Calls `extract_email_text()` to parse each one properly (handles MIME, multipart, HTML)
- Anonymizes all personal information:
  - Email addresses → `[email]`
  - URLs → `[url]`
  - Phone numbers → `[phone]`
  - Names → `[name]`
- Saves all extracted emails with label `"spam"` to `data/anonymized_spam.csv`
- This CSV is then automatically merged into training data the next time you run `train_model.py`

> Note: The name list used for anonymization in the current code is `["adhipatya", "saxena", "adhip"]`. Update this list in `parse_mbox.py` with your own names before running it.

---

## 📦 Module Reference

### `src/data_loader.py`

Loads the spam CSV dataset, renames columns to `label` and `text`, and maps `"ham" → 0`, `"spam" → 1`.

```python
df = load_data("data/spam.csv")
```

---

### `src/preprocess.py`

Cleans raw text: lowercase → remove non-alphanumeric characters → remove NLTK English stopwords.

```python
cleaned = clean_text(raw_text)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Raw input text |

**Returns:** A cleaned string with only meaningful words.

---

### `src/train_model.py`

Trains Naive Bayes and Logistic Regression pipelines on the dataset, compares accuracy, and saves the better model as `models/spam_pipeline.pkl`.

```bash
python src/train_model.py
```

---

### `src/evaluate_model.py`

Loads the saved model and evaluates it on the test split. Prints accuracy, precision, recall, F1 score, full classification report, and shows a confusion matrix heatmap.

```bash
python src/evaluate_model.py
```

---

### `src/predict.py`

The core prediction module. Contains four main components:

| Component | What It Does |
|-----------|-------------|
| `EmailHTMLStripper` | HTMLParser subclass that strips HTML tags, tracks hidden `display:none` content for Bayesian poisoning detection |
| `parse_email_message(raw)` | Parses raw MIME/SMTP email string → extracts subject, body, from field; handles plain text and HTML emails |
| `check_heuristics(subject, body, hidden_len)` | Checks 15 phishing patterns and Bayesian poisoning; returns `(is_spam, confidence)` |
| `predict_message(message)` | Full pipeline: parse → heuristics → ML → returns `(prediction, confidence_%)` |

```python
prediction, confidence = predict_message(message)
# prediction: 1 = spam, 0 = ham
# confidence: float, e.g. 97.5
```

---

### `src/parse_mbox.py`

Reads `data/Spam.mbox`, extracts each email using `extract_email_text()`, anonymizes personal info (emails, URLs, phones, names), and saves all entries to `data/anonymized_spam.csv` with label `"spam"`.

```bash
python src/parse_mbox.py
```

---

### `src/wordcloud_visualization.py`

Loads and cleans the dataset, joins all spam text and all ham text separately, and generates two word cloud images.

```bash
python src/wordcloud_visualization.py
```

---

### `src/utils.py`

General-purpose helper functions used across the project.

| Function | Description |
|----------|-------------|
| `create_directory(path)` | Creates a directory if it doesn't already exist |
| `print_separator()` | Prints a `===` divider line to the terminal |
| `print_title(title)` | Prints a formatted uppercase title surrounded by separators |

---

## ⚠️ Known Limitations

| Issue | Details |
|-------|---------|
| **SMS-based training data** | The base `spam.csv` dataset is made up of short SMS messages. Long-form phishing emails with complex formatting may behave differently from what the model was trained on. |
| **Heuristics are hardcoded** | The 15 phishing phrases in `check_heuristics()` are fixed in the source code. New phishing tactics that use different wording won't be caught unless the list is updated manually. |
| **Name list for anonymization** | The `anonymize_text()` function in `parse_mbox.py` uses a hardcoded list of names. You must edit this list before running it on your own mailbox. |
| **Confidence scores are not calibrated** | The confidence percentage comes from raw `predict_proba()` values which may not be perfectly calibrated. A 90% confidence does not necessarily mean the model is exactly 90% sure. |
| **No multi-language support** | The preprocessing and stopword removal are English-only. Spam in other languages will not be processed correctly. |
| **No persistent history** | The Streamlit app does not save any scan results. Each session starts fresh. |
| **Evaluation uses only the base dataset** | `evaluate_model.py` evaluates on `spam.csv` only. If you added `anonymized_spam.csv` during training, the evaluation doesn't include that extra data. |

---

## 🗺️ Roadmap

- [x] SMS spam dataset loading and preprocessing
- [x] TF-IDF vectorization inside a sklearn Pipeline
- [x] Naive Bayes classifier
- [x] Logistic Regression classifier
- [x] Automatic best model selection and saving
- [x] Model evaluation (accuracy, precision, recall, F1, confusion matrix)
- [x] Heuristic detection layer (15 phishing phrases, urgency detection)
- [x] Bayesian poisoning detection (hidden `display:none` text measurement)
- [x] Full MIME/SMTP email parser
- [x] Custom `EmailHTMLStripper` with hidden text tracking
- [x] `.mbox` mailbox parser with anonymization
- [x] Custom anonymized spam dataset from real emails
- [x] Spam and ham word cloud generation
- [x] Streamlit web app (live scan, risk levels, file upload, view mode toggle)
- [x] CLI menu interface (Train / Evaluate / Predict)
- [x] Custom dark red Streamlit theme
- [x] Export evaluation report as a downloadable PDF
- [x] Deploy the Streamlit app on Streamlit Cloud
- [ ] Add more ML models (Random Forest, SVM, XGBoost) to the comparison
- [ ] Add URL reputation checking for links found inside messages
- [ ] Expand heuristic phrases list and make it configurable via a JSON file
- [ ] Save scan history per session in the Streamlit app
- [ ] Add multi-language spam detection support

---

<div align="center">

Built with ❤️ using Python, Scikit-Learn, NLTK, and Streamlit.

</div>
