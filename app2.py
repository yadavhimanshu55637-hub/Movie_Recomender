# Streamlit movie recommendation app. Run with:
#   streamlit run app2.py

import os
import sys
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MOVIES_CSV = "data/movies.csv"


def create_sample_dataset(csv_path: str = MOVIES_CSV) -> None:
    """Creates a dummy dataset if the file doesn't exist to prevent crashes."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sample_data = {
        "title": ["Inception", "Interstellar", "The Matrix", "Toy Story", "The Dark Knight"],
        "description": [
            "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.",
            "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
            "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom.",
            "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests."
        ]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)


@st.cache_data
def load_movies(csv_path: str = MOVIES_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        create_sample_dataset(csv_path)
    return pd.read_csv(csv_path)


@st.cache_data
def build_tfidf_matrix(movies: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["description"])
    return vectorizer, tfidf_matrix


def recommend_movies(title: str, movies: pd.DataFrame, tfidf_matrix, top_n: int = 5, min_score: float = 0.0) -> pd.DataFrame:
    if title not in movies["title"].values:
        raise ValueError(f"Movie '{title}' not found in the dataset.")

    idx = movies.index[movies["title"] == title][0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != idx]

    recommended = movies.iloc[similar_indices].copy()
    recommended["score"] = cosine_similarities[similar_indices]
    recommended = recommended[recommended["score"] >= min_score]
    return recommended[["title", "description", "score"]].head(top_n)


def search_movies(query: str, movies: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix) -> pd.DataFrame:
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix).flatten()
    results = movies.copy()
    results["score"] = cosine_similarities
    matches = results[results["title"].str.contains(query, case=False, na=False) |
                      results["description"].str.contains(query, case=False, na=False)]
    return matches.sort_values("score", ascending=False)


def extract_popular_tags(movies: pd.DataFrame, top_n: int = 10) -> list[str]:
    token_counts: dict[str, int] = {}
    for desc in movies["description"]:
        for token in desc.split():
            token = token.strip().lower()
            if len(token) < 4:
                continue
            token_counts[token] = token_counts.get(token, 0) + 1
    common = sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [tag for tag, _ in common[:top_n]]


def render_movie_cards(movies: pd.DataFrame, show_description: bool = True) -> None:
    for _, row in movies.iterrows():
        with st.expander(f"{row['title']} ({row['score']:.2f})"):
            if show_description:
                st.write(row["description"])


def render_data_table(movies: pd.DataFrame) -> None:
    st.dataframe(movies.reset_index(drop=True), use_container_width=True)


def app() -> None:
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommendation System")
    st.write(
        "Explore a content-based recommendation engine powered by TF-IDF similarity on movie descriptions."
    )

    # Check and load data safely
    if not os.path.exists(MOVIES_CSV):
        st.warning("⚠️ `data/movies.csv` was missing. A sample dataset has been generated automatically so the app works out-of-the-box!")
    
    movies = load_movies()
    vectorizer, tfidf_matrix = build_tfidf_matrix(movies)
    tags = extract_popular_tags(movies)

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Choose a view", ["Recommend", "Search", "Explore", "About"])
        st.markdown("---")
        st.write("**Quick filters**")
        selected_tag = st.selectbox("Browse by keyword", ["All"] + tags)
        st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="min_score")
        st.markdown("---")
        st.write("Need more data? Add movie rows into `data/movies.csv`.")

    if page == "Recommend":
        st.subheader("Movie-based Recommendations")
        selected_movie = st.selectbox("Choose a movie", movies["title"].tolist())
        top_n = st.slider("How many recommendations?", min_value=3, max_value=12, value=5)
        min_score = st.session_state.min_score
        show_description = st.checkbox("Show descriptions", value=True)

        try:
            recommendations = recommend_movies(selected_movie, movies, tfidf_matrix, top_n=top_n, min_score=min_score)
            if recommendations.empty:
                st.warning("No movies matched the selected similarity threshold. Lower the minimum score or choose another movie.")
            else:
                render_movie_cards(recommendations, show_description=show_description)
        except ValueError as exc:
            st.error(str(exc))

        st.markdown("---")
        st.write("**Movie details**")
        detail_row = movies[movies["title"] == selected_movie].iloc[0]
        st.write(f"**Description:** {detail_row['description']}")
        tag_list = [token for token in detail_row["description"].split() if len(token) > 3]
        st.write("**Key tags:**", ", ".join(tag_list))

    elif page == "Search":
        st.subheader("Keyword Search")
        query = st.text_input("Search movie titles or descriptions", value="space adventure")
        show_description = st.checkbox("Show descriptions", value=True, key="search_desc")

        if query:
            results = search_movies(query, movies, vectorizer, tfidf_matrix)
            if results.empty:
                st.warning("No search results found for your query.")
            else:
                st.write(f"Found {len(results)} matching movies.")
                if show_description:
                    render_movie_cards(results, show_description=True)
                else:
                    render_data_table(results[["title", "score"]])

    elif page == "Explore":
        st.subheader("Dataset Explorer")
        if selected_tag != "All":
            filtered = movies[movies["description"].str.contains(selected_tag, case=False, na=False)]
            st.write(f"Showing movies containing '{selected_tag}'.")
        else:
            filtered = movies
            st.write("Showing all movies.")

        render_data_table(filtered[["title", "description"]])

    else:
        st.subheader("About this project")
        st.markdown(
            """
            - Uses TF-IDF vectorization to convert movie descriptions into numerical features.
            - Finds similar movies using cosine similarity.
            - Supports searching by keyword and filtering by genre-related tags.
            - Built with Streamlit for an interactive web interface.
            """
        )
        st.write("Add more movies to `data/movies.csv` to improve the recommendations.")
        st.write("Run with: `streamlit run app2.py`")


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        app()
    else:
        st.write("Please run this app via Streamlit: `streamlit run app2.py`")