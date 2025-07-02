import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("D:/LPU/Interns/Avnotech/Task 1/movies.csv")
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['content'] = movies['title'] + ' ' + movies['genres']
    return movies

movies = load_data()

# Vectorize content
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(movie_title):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("Enter a movie title to get similar recommendations:")

# Show a few examples to guide the user
st.info("👉 **Example titles:** Toy Story (1995), Jumanji (1995), Heat (1995), GoldenEye (1995), Casino (1995)")

# Optional: Dropdown of sample titles
sample_titles = ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'GoldenEye (1995)', 'Casino (1995)']
selected_movie = st.selectbox("Or choose a movie from the list:", sample_titles)
custom_input = st.text_input("Or type your own movie title:")

# Decide final input
movie_input = custom_input if custom_input else selected_movie

if st.button("Recommend"):
    if movie_input:
        results = recommend(movie_input)
        if results:
            st.subheader(f"Top 5 Recommendations for '{movie_input}':")
            for i, movie in enumerate(results, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error("❌ Movie not found. Please check the title and try again.")
    else:
        st.warning("⚠️ Please enter or select a movie title.")
