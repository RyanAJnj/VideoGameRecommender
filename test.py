import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Session state initialization
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Cards View"

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load CSV with cache
@st.cache_data
def load_data():
    return pd.read_csv("games.csv")

# Clean text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# Sentiment
def get_sentiment(text):
    if isinstance(text, str):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 1
        elif score <= -0.05:
            return 0
    return None

# Model and encoder cache
@st.cache_resource
def prepare_model(df):
    df.drop_duplicates(inplace=True)
    df['Title'] = df['Title'].astype(str).str.title()
    df['Summary'] = df['Summary'].astype(str).apply(clean_text)
    
    for col in ['Genres', 'Team', 'Reviews']:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [])
    
    df['reviews_text'] = df['Reviews'].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    df['liked'] = df['reviews_text'].apply(get_sentiment)
    df_filtered = df[df['liked'].isin([0, 1])].copy()

    genre_encoder = MultiLabelBinarizer()
    team_encoder = MultiLabelBinarizer()
    df_genres = pd.DataFrame(genre_encoder.fit_transform(df_filtered['Genres']), columns=genre_encoder.classes_)
    df_teams = pd.DataFrame(team_encoder.fit_transform(df_filtered['Team']), columns=team_encoder.classes_)

    tfidf = TfidfVectorizer(max_features=300)
    df_filtered['Summary'] = df_filtered['Summary'].fillna("")
    summary_tfidf = tfidf.fit_transform(df_filtered['Summary'])
    df_summary = pd.DataFrame(summary_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    X_dense = pd.concat([df_genres, df_teams, df_summary], axis=1)
    X_sparse = csr_matrix(X_dense.values)
    y = df_filtered['liked']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_sparse, y)
    df_filtered['like_probability'] = model.predict_proba(X_sparse)[:, 1]

    return model, genre_encoder, team_encoder, tfidf, df_filtered, X_sparse, X_dense.columns.tolist()

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Load everything
df = load_data()
model, genre_encoder, team_encoder, tfidf, df_filtered, X_sparse, feature_names = prepare_model(df)

# UI
st.title("🎮 Personalized Video Game Recommender")

user_genres = st.sidebar.multiselect("Select Genres You Like:", genre_encoder.classes_)
user_teams = st.sidebar.multiselect("Select Developers/Publishers You Like:", team_encoder.classes_)
user_text = st.text_area("Briefly describe what kinds of games you enjoy:")
submit = st.button("🔍 Get My Recommendations")

if submit:
    if not (user_genres or user_teams or user_text.strip()):
        st.warning("Please provide some input to get recommendations.")
    else:
        user_vector = pd.DataFrame(columns=feature_names)
        user_vector.loc[0] = 0

        for genre in user_genres:
            if genre in user_vector.columns:
                user_vector.loc[0, genre] = 1

        for team in user_teams:
            if team in user_vector.columns:
                user_vector.loc[0, team] = 1

        if user_text.strip():
            user_text_vector = tfidf.transform([user_text])
            tfidf_feature_names = tfidf.get_feature_names_out()
            user_text_df = pd.DataFrame(user_text_vector.toarray(), columns=tfidf_feature_names)
            for col in user_text_df.columns:
                if col in user_vector.columns:
                    user_vector.loc[0, col] = user_text_df.loc[0, col]
            user_combined_vector = csr_matrix(user_vector.values)
        else:
            user_combined_vector = csr_matrix(user_vector.values)

        similarities = cosine_similarity(user_combined_vector, X_sparse)[0]
        df_filtered["similarity"] = similarities

        top_recommendations = df_filtered.sort_values(by="similarity", ascending=False).head(30)
        top_unique = top_recommendations.drop_duplicates(subset='Title', keep='first')
        top_unique = top_unique.sort_values(by='like_probability', ascending=True).tail(10)
        st.session_state.recommendations = top_unique

        # Cards View - Top Section
        st.write("### Card View")
        cols = st.columns(2)
        for idx, (_, row) in enumerate(top_unique.sort_values(by="like_probability", ascending=False).iterrows()):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(row['Title'])
                    st.caption(f"**Genres:** {', '.join(row['Genres'])}")
                    st.caption(f"**Team:** {', '.join(row['Team'])}")
                    prob = row['like_probability']
                    st.progress(int(prob * 100), text=f"Like probability: {prob:.1%}")
                    st.metric("Match score", f"{row['similarity']:.2f}", delta_color="off")
        
        # Table View - Bottom Section 
        st.write("### Table View")
        display_df = top_unique[['Title', 'Genres', 'Team', 'like_probability', 'similarity']].copy()
        display_df['Genres'] = display_df['Genres'].apply(lambda x: ', '.join(x))
        display_df['Team'] = display_df['Team'].apply(lambda x: ', '.join(x))
        display_df.rename(columns={
            'Title': 'Game Title',
            'like_probability': 'Like %',
            'similarity': 'Match Score'
        }, inplace=True)
        
        st.dataframe(
            display_df.sort_values('Like %', ascending=False),
            column_config={
                "Like %": st.column_config.ProgressColumn(
                    "Like %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Match Score": st.column_config.NumberColumn(
                    format="%.2f"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=400  # Fixed height for better layout
        )
