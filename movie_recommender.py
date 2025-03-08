import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import time

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color schemes
light_theme = {
    "bg_color": "#ffffff",
    "text_color": "#1f1f1f",
    "accent_color": "#1f77b4",
    "secondary_color": "#ff9d00",
    "card_bg": "#f0f2f6"
}

dark_theme = {
    "bg_color": "#0e1117",
    "text_color": "#fafafa",
    "accent_color": "#00b4d8",
    "secondary_color": "#ffd166",
    "card_bg": "#1e2130"
}

# Session state initialization
if 'theme' not in st.session_state:
    st.session_state.theme = dark_theme

if 'history' not in st.session_state:
    st.session_state.history = []

# Apply custom CSS based on theme
def apply_theme(theme):
    st.markdown(f"""
    <style>
    .main {{
        background-color: {theme["bg_color"]};
        color: {theme["text_color"]};
        padding: 2rem;
    }}
    .title {{
        text-align: center;
        color: {theme["accent_color"]};
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    .subtitle {{
        color: {theme["secondary_color"]};
        font-size: 1.5rem;
    }}
    .stButton>button {{
        background-color: {theme["accent_color"]};
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {theme["secondary_color"]};
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }}
    .movie-card {{
        background-color: {theme["card_bg"]};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }}
    .movie-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }}
    .stExpander {{
        background-color: {theme["card_bg"]};
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply theme
apply_theme(st.session_state.theme)

# Sidebar
with st.sidebar:
    st.markdown(f"<h2 style='color: {st.session_state.theme['accent_color']}'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    # Theme toggle
    theme_option = st.radio("Choose Theme", ["Dark Mode", "Light Mode"])
    if theme_option == "Dark Mode" and st.session_state.theme != dark_theme:
        st.session_state.theme = dark_theme
        st.rerun()
    elif theme_option == "Light Mode" and st.session_state.theme != light_theme:
        st.session_state.theme = light_theme
        st.rerun()
    
    st.markdown("---")
    
    # History section
    st.markdown(f"<h3 style='color: {st.session_state.theme['accent_color']}'>🕒 Recent Searches</h3>", unsafe_allow_html=True)
    
    if st.session_state.history:
        for i, movie in enumerate(st.session_state.history[-5:]):
            if st.button(f"🔍 {movie}", key=f"history_{i}"):
                st.session_state.selected_movie = movie
                st.rerun()
    else:
        st.write("No recent searches")
    
    st.markdown("---")
    
    # About section
    st.markdown(f"<h3 style='color: {st.session_state.theme['accent_color']}'>ℹ️ About</h3>", unsafe_allow_html=True)
    st.markdown("""
    This movie recommender system uses content-based filtering to suggest similar movies based on:
    - Movie genres
    - Director
    - Main actors
    
    The system calculates similarity between movies using cosine similarity on these features.
    """)

# Title
st.markdown("<h1 class='title'>🎬 CineMatch</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='text-align: center;'>Your Personal Movie Recommendation Engine</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Load the movie data
    df = pd.read_csv('datasets/final_data.csv')
    return df

@st.cache_data
def load_full_data():
    # Load the full movie data with more details
    df = pd.read_csv('datasets/movie_metadata.csv')
    return df

def get_recommendations(title, cosine_sim, df, num_recommendations=10):
    # Get the index of the movie that matches the title
    idx = df[df['movie_title'].str.lower() == title.lower()].index[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the most similar movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top most similar movies
    return df['movie_title'].iloc[movie_indices], [score[1] for score in sim_scores]

def get_movie_details(title, full_df):
    # Get details for a specific movie
    movie = full_df[full_df['movie_title'].str.lower() == title.lower()]
    if not movie.empty:
        return movie.iloc[0]
    return None

def get_movie_poster_url(title):
    # This is a placeholder function - in a real app, you would use a movie API
    # to get actual poster URLs. For now, we'll return a placeholder image.
    return f"https://via.placeholder.com/300x450.png?text={title.replace(' ', '+')}"

def filter_by_genre(df, genre):
    # Filter movies by genre
    return df[df['genres'].str.contains(genre, case=False)]

def create_radar_chart(movie_details):
    # Create a radar chart for movie attributes
    categories = ['IMDb Score', 'Budget (M)', 'Gross (M)', 'Facebook Likes']
    
    # Normalize values for radar chart
    imdb = float(movie_details.get('imdb_score', 5)) / 10
    budget = float(movie_details.get('budget', 100000000)) / 300000000
    gross = float(movie_details.get('gross', 100000000)) / 800000000
    likes = float(movie_details.get('movie_facebook_likes', 1000)) / 200000
    
    values = [imdb, budget, gross, likes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=movie_details.get('movie_title', 'Movie'),
        line=dict(color=st.session_state.theme['accent_color'])
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=st.session_state.theme['text_color'])
    )
    
    return fig

# Main app functionality
try:
    # Load data
    df = load_data()
    full_df = load_full_data()
    
    # Create count matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['comb'])
    
    # Compute the Cosine Similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Extract all unique genres
    all_genres = []
    for genres in df['genres'].str.split():
        all_genres.extend(genres)
    unique_genres = sorted(list(set(all_genres)))
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔍 Find Recommendations", "🎭 Browse by Genre", "📊 Movie Analytics"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a list of all movie titles
            all_titles = df['movie_title'].tolist()
            
            # Create the main interface
            st.markdown("### Select a Movie")
            
            # Initialize session state for selected movie if not exists
            if 'selected_movie' not in st.session_state:
                st.session_state.selected_movie = all_titles[0]
            
            # Movie selection
            selected_movie = st.selectbox(
                "Type or select a movie from the dropdown",
                all_titles,
                index=all_titles.index(st.session_state.selected_movie) if st.session_state.selected_movie in all_titles else 0,
                key="movie_selector"
            )
            
            # Update session state
            st.session_state.selected_movie = selected_movie
            
            # Add to history if not already the most recent
            if not st.session_state.history or st.session_state.history[-1] != selected_movie:
                st.session_state.history.append(selected_movie)
                # Keep only the last 10 searches
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)
            
            # Number of recommendations slider
            num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
            
            # Get recommendations button
            if st.button('Get Recommendations', key='get_recs_button'):
                with st.spinner('Finding similar movies...'):
                    # Get recommendations
                    recommended_movies, similarity_scores = get_recommendations(
                        selected_movie, 
                        cosine_sim,
                        df,
                        num_recommendations
                    )
                    
                    # Store in session state
                    st.session_state.recommendations = list(zip(recommended_movies, similarity_scores))
                    
                    # Success message with animation
                    st.success(f"Found {len(recommended_movies)} movies similar to '{selected_movie}'!")
        
        with col2:
            # Display selected movie details
            movie_details = get_movie_details(selected_movie, full_df)
            
            if movie_details is not None:
                st.markdown(f"### Movie Details")
                
                # Movie poster (placeholder)
                st.image(get_movie_poster_url(selected_movie), width=200)
                
                # Movie info
                st.markdown(f"**Director:** {movie_details.get('director_name', 'Unknown')}")
                st.markdown(f"**Year:** {movie_details.get('title_year', 'Unknown')}")
                st.markdown(f"**Genre:** {movie_details.get('genres', 'Unknown')}")
                st.markdown(f"**IMDb Score:** {movie_details.get('imdb_score', 'Unknown')}/10")
                
                # Cast
                st.markdown("**Cast:**")
                cast = [
                    movie_details.get('actor_1_name', ''),
                    movie_details.get('actor_2_name', ''),
                    movie_details.get('actor_3_name', '')
                ]
                cast = [actor for actor in cast if actor]
                st.markdown(", ".join(cast))
        
        # Display recommendations if available
        if 'recommendations' in st.session_state:
            st.markdown("---")
            st.markdown("### Recommended Movies")
            
            # Create a visualization of similarity scores
            fig = px.bar(
                x=[movie for movie, _ in st.session_state.recommendations],
                y=[score for _, score in st.session_state.recommendations],
                labels={'x': 'Movie', 'y': 'Similarity Score'},
                title='Movie Similarity Scores',
                color=[score for _, score in st.session_state.recommendations],
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=st.session_state.theme['text_color'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display movie cards
            cols = st.columns(3)
            for i, (movie, score) in enumerate(st.session_state.recommendations):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class='movie-card'>
                            <h4>{movie}</h4>
                            <p>Similarity: {score:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Details for {movie}", key=f"details_{i}"):
                            st.session_state.selected_movie = movie
                            st.rerun()
    
    with tab2:
        st.markdown("### Browse Movies by Genre")
        
        # Genre selection
        selected_genre = st.selectbox("Select a genre", unique_genres)
        
        # Filter movies by genre
        filtered_movies = filter_by_genre(df, selected_genre)
        
        if not filtered_movies.empty:
            st.success(f"Found {len(filtered_movies)} movies in the {selected_genre} genre!")
            
            # Display movies in a grid
            cols = st.columns(3)
            for i, (_, row) in enumerate(filtered_movies.iterrows()):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class='movie-card'>
                            <h4>{row['movie_title']}</h4>
                            <p>Director: {row['director_name']}</p>
                            <p>Cast: {row['actor_1_name']}, {row['actor_2_name']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Select {row['movie_title']}", key=f"select_{i}"):
                            st.session_state.selected_movie = row['movie_title']
                            st.rerun()
        else:
            st.warning(f"No movies found in the {selected_genre} genre.")
    
    with tab3:
        st.markdown("### Movie Analytics")
        
        if 'selected_movie' in st.session_state:
            movie_details = get_movie_details(st.session_state.selected_movie, full_df)
            
            if movie_details is not None:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"### {movie_details.get('movie_title', 'Movie')} Stats")
                    
                    # Radar chart
                    radar_fig = create_radar_chart(movie_details)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Movie Details")
                    
                    # Additional details
                    details = {
                        "Budget": f"${int(float(movie_details.get('budget', 0))/1000000)}M",
                        "Gross Revenue": f"${int(float(movie_details.get('gross', 0))/1000000)}M",
                        "IMDb Score": f"{movie_details.get('imdb_score', 'N/A')}/10",
                        "Content Rating": movie_details.get('content_rating', 'N/A'),
                        "Duration": f"{movie_details.get('duration', 'N/A')} minutes",
                        "Language": movie_details.get('language', 'N/A'),
                        "Country": movie_details.get('country', 'N/A'),
                        "Year": movie_details.get('title_year', 'N/A')
                    }
                    
                    for key, value in details.items():
                        st.markdown(f"**{key}:** {value}")
                    
                    # Plot keywords
                    if 'plot_keywords' in movie_details and movie_details['plot_keywords']:
                        st.markdown("**Plot Keywords:**")
                        keywords = str(movie_details['plot_keywords']).split('|')
                        st.markdown(", ".join([f"`{keyword}`" for keyword in keywords]))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure the dataset files exist in the correct location.") 