# 🎮 Personalized AI Video Game Recommender

An interactive recommendation system built with machine learning and natural language processing (NLP) to help users discover video games they'll enjoy based on their preferences and interests.

## 🚀 Features

- 🧠 **AI Model**: Trained a logistic regression classifier using TF-IDF features and user sentiment labels.
- ✍️ **Text Analysis**: Extracts user intent using free-form text input.
- 📊 **Cosine Similarity**: Compares user preferences with game features for similarity scoring.
- 📚 **NLP Pipeline**: Cleans and processes summaries and user reviews using NLTK and BeautifulSoup.
- 💬 **Sentiment Analysis**: Applies VADER to user reviews to auto-label games as liked/disliked.
- 🖥️ **Streamlit UI**: Simple, responsive app where users can select genres, companies, and describe their gaming interests.
- 📈 **Recommendations**: Combines similarity and predicted like probability to show top 10 game recommendations.

---

## 🛠 Tech Stack

- **Language**: Python 3
- **Framework**: Streamlit
- **ML Tools**: Scikit-learn (Logistic Regression, TF-IDF, Cosine Similarity)
- **NLP**: NLTK, VADER Sentiment Analyzer, BeautifulSoup
- **Data**: CSV dataset of video games with summaries, genres, teams, reviews, and more


---

## 🧪 How It Works

1. **Data Preprocessing**:
   - Cleans summaries and reviews
   - Parses genres and team data
   - Applies sentiment analysis to generate labels

2. **Model Training**:
   - Multi-label encodes genres and teams
   - Vectorizes summaries using TF-IDF
   - Trains logistic regression to predict game likability

3. **User Input**:
   - Users select preferred genres, teams, and describe what games they like

4. **Recommendation Engine**:
   - Transforms user input into a vector
   - Computes cosine similarity with all games
   - Filters and ranks top recommendations

---

## 🧑‍💻 Running Locally

1. Clone the repository:
```bash
git clone https://github.com/RyanAJnj/videogame-recommender.git
cd videogame-recommender


