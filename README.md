# 🎭 Predict the Mood
### Sentiment Analysis using Machine Learning Models with YouTube Comment Scraping

> **Published in:** IJARESM, Volume 12 Issue 12, December 2024 | Impact Factor: 8.536  
> **Author:** Satyam Shah — VIT Bhopal University  
> **Certificate ID:** IJ-0512241222

---

## 📌 Overview

**Predict the Mood** is a machine learning–powered web application that performs **sentiment analysis on YouTube comments** in real time. Given a YouTube video URL, the app scrapes its comments and classifies each one as **Positive**, **Negative**, or **Neutral** using multiple pre-trained ML models. The results are visualized to give creators and researchers a quick pulse on audience sentiment.

This project was developed as part of a research study comparing the effectiveness of classical machine learning algorithms on NLP tasks — specifically, public opinion mining from social media platforms.

---

## 🧠 How It Works

1. **Input** — User provides a YouTube video URL via the web interface.
2. **Scraping** — The app uses the YouTube Data API (or `google-api-python-client`) to fetch comments from the video.
3. **Preprocessing** — Raw comments are cleaned (lowercased, punctuation removed, stopwords filtered).
4. **Vectorization** — Text is converted to numerical features using a **TF-IDF** vectorizer (`tfidf_model.pkl`).
5. **Prediction** — Four ML models classify the sentiment of each comment:
   - Logistic Regression (`lr_model.pkl`)
   - Naive Bayes (`nb_model.pkl`)
   - Decision Tree (`dt_model.pkl`)
   - K-Nearest Neighbors (`knn_model.pkl`)
6. **Output** — Sentiment results and distribution charts are displayed on the UI.

---

## 🗂️ Project Structure

```
Predict-the-Mood/
│
├── deploy.py              # Main Streamlit application
├── tfidf_model.pkl        # Pre-trained TF-IDF vectorizer
├── lr_model.pkl           # Logistic Regression model
├── nb_model.pkl           # Naive Bayes model
├── dt_model.pkl           # Decision Tree model
├── knn_model.pkl          # K-Nearest Neighbors model
├── youtube.png            # UI asset / logo
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🤖 Models Used

| Model | Type | Strengths |
|---|---|---|
| Logistic Regression | Linear Classifier | Fast, interpretable, strong baseline |
| Naive Bayes | Probabilistic | Excellent for text classification |
| Decision Tree | Tree-based | Explainable, handles non-linearity |
| K-Nearest Neighbors | Instance-based | Simple, no training phase assumptions |

All models were trained on a labeled sentiment dataset and serialized using **Pickle** for fast inference at runtime.

---

## ⚙️ Prerequisites

Before running locally, make sure you have:

- **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- A **YouTube Data API v3 key** from [Google Cloud Console](https://console.cloud.google.com/)
- `pip` package manager

---

## 🚀 Running Locally

### Step 1 — Clone the Repository

```bash
git clone https://github.com/satyams8092/Predict-the-Mood.git
cd Predict-the-Mood
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set Up Your YouTube API Key

The app requires a **YouTube Data API v3** key to scrape comments.

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Navigate to **APIs & Services → Library**
4. Search for **YouTube Data API v3** and enable it
5. Go to **APIs & Services → Credentials → Create Credentials → API Key**
6. Copy your API key

You can set it as an environment variable:

```bash
# On Windows (Command Prompt)
set YOUTUBE_API_KEY=your_api_key_here

# On macOS/Linux
export YOUTUBE_API_KEY=your_api_key_here
```

Or paste it directly into `deploy.py` where the API key is referenced (look for a variable like `api_key` or `API_KEY`).

### Step 5 — Run the App

```bash
streamlit run deploy.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🖥️ Usage

1. Launch the app using the command above.
2. Paste a **YouTube video URL** into the input field (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
3. Select a **machine learning model** from the available options.
4. Click **Analyze** (or the equivalent button).
5. View the **sentiment breakdown** — positive, negative, and neutral comment counts along with visual charts.

---

## 📦 Dependencies

Key libraries used in this project (from `requirements.txt`):

| Library | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `google-api-python-client` | YouTube Data API integration |
| `scikit-learn` | ML models and TF-IDF vectorizer |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `nltk` | Natural language preprocessing |
| `matplotlib` / `seaborn` | Data visualization |
| `pickle` | Model serialization/loading |

Install all at once with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

**`ModuleNotFoundError`** — Make sure your virtual environment is activated and all dependencies are installed.

**`quota exceeded` or API errors** — YouTube Data API has a daily quota limit (10,000 units). Check your usage in Google Cloud Console.

**`streamlit: command not found`** — Try running with `python -m streamlit run deploy.py` instead.

**Comments not loading** — Ensure comments are enabled on the YouTube video and your API key has the correct permissions.

---

## 📄 Research Publication

This project was published as a research paper:

> **"Predict the Mood: Sentiment Analysis using Machine Learning Models with YouTube Comment Scraping"**  
> Satyam Shah, VIT Bhopal University  
> *IJARESM*, Volume 12, Issue 12, December 2024  
> ISSN: 2455-6211 | Impact Factor: 8.536 | UGC Journal No. 7647

---

## 👤 Author

**Satyam Shah**  
Student, School of Computing Science Engineering and Artificial Intelligence  
VIT Bhopal University, Madhya Pradesh – 466114  

- GitHub: [@satyams8092](https://github.com/satyams8092)

---

## 📝 License

This project is open-source. Feel free to fork, star ⭐, and build upon it!

---

> *"I programmed it for 14 hours straight :\"* — Satyam Shah
