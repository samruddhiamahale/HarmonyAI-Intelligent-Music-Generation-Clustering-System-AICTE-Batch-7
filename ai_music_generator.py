import streamlit as st
from google import genai
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



# Load Environment

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

HISTORY_FILE = "music_history.json"



# Load Dataset from CSV

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset.csv")
    df = df.fillna("")  # prevent errors if any empty cells
    return df

df = load_dataset()



# Convert to list format (so your old ML code still works)
song_dataset = df.to_dict(orient="records")



# Load History
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        saved_history = json.load(f)
else:
    saved_history = []

if "history" not in st.session_state:
    st.session_state.history = saved_history

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.history, f, indent=4)



# Gemini Generator
def generate_with_retry(prompt, temperature):
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": temperature,
            }
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"



# Cover Art Generator
def generate_cover_art(song_text):
    prompt = f"""
    Create a detailed and creative album cover art description 
    for the following song:

    {song_text}

    Describe colors, background, lighting, mood and artistic style.
    """
    return generate_with_retry(prompt, 0.8)



# Helper Functions
def loading_animation():
    with st.spinner("Generating with AI..."):
        time.sleep(1)

def song_structure_visualizer():
    st.subheader("🎼 Song Structure Visualizer")
    structure = ["Verse", "Chorus", "Verse", "Bridge", "Final Chorus"]
    fig, ax = plt.subplots(figsize=(8, 1))
    for i in range(len(structure)):
        ax.barh(0, 1, left=i)
    ax.set_xlim(0, len(structure))
    ax.set_yticks([])
    ax.set_xticks(range(len(structure)))
    ax.set_xticklabels(structure, rotation=45)
    st.pyplot(fig)

def save_output(title, content):
    entry = {
        "title": title,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cover_art": None
    }
    st.session_state.history.insert(0, entry)
    save_history()




# AI/ML - Audio Feature Clustering
@st.cache_data
def train_kmeans():
    df_copy = df.copy()

    feature_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    # Convert safely to numeric
    for col in feature_columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    df_copy[feature_columns] = df_copy[feature_columns].fillna(0)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_copy["cluster"] = kmeans.fit_predict(df_copy[feature_columns])

    return df_copy

def recommend_songs_by_features(selected_song):
    clustered_df = train_kmeans()

    if selected_song not in clustered_df["track_name"].values:
        return []

    selected_cluster = clustered_df.loc[
        clustered_df["track_name"] == selected_song, "cluster"
    ].iloc[0]

    recommendations = clustered_df[
        clustered_df["cluster"] == selected_cluster
    ]

    recommendations = recommendations[
        recommendations["track_name"] != selected_song
    ]

    return recommendations[["track_name", "artists"]].head(5)




# Page Config
st.set_page_config(
    page_title="HarmonyAI",
    page_icon="🎵",
    layout="wide"
)




# Hero Section
st.markdown("""
<div style='text-align: center; padding: 40px;'>
    <h1>🎵 HarmonyAI</h1>
    <p style='font-size:20px; color:#333;'>
        Intelligent Music Generation & Clustering Platform
    </p>
    <p style='font-size:16px; color:#333;'>
        Powered by Generative AI & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)



# Sidebar Controls
st.sidebar.markdown("""
<h2 style='color:#FF4B4B;'>⚙️ HarmonyAI Controls</h2>
<hr>
""", unsafe_allow_html=True)

producer_mode = st.sidebar.selectbox(
    "🎙 AI Producer Personality",
    ["Classic Producer", "Modern Hitmaker", "Underground Indie Creator", "Bollywood Composer"]
)

collab_artist = st.sidebar.text_input("🤝 Collaboration Artist (Optional)")
temperature = st.sidebar.slider("🎨 Creativity Level", 0.2, 1.0, 0.7)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Platform Stats")

total_songs = len(df)
ai_features = 6  # you have 6 generation features
clusters = 5     # KMeans n_clusters=5

st.sidebar.markdown(
    f"""
    <div style="background-color:#1f2937;padding:15px;border-radius:10px;margin-bottom:10px;">
        <h4 style="color:white;margin:0;">🎵 Total Songs</h4>
        <p style="font-size:22px;color:#60a5fa;margin:0;">{total_songs}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="background-color:#1f2937;padding:15px;border-radius:10px;margin-bottom:10px;">
        <h4 style="color:white;margin:0;">🤖 AI Features</h4>
        <p style="font-size:22px;color:#34d399;margin:0;">{ai_features}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="background-color:#1f2937;padding:15px;border-radius:10px;">
        <h4 style="color:white;margin:0;">📊 Clusters</h4>
        <p style="font-size:22px;color:#fbbf24;margin:0;">{clusters}</p>
    </div>
    """,
    unsafe_allow_html=True
)



# TOP NAVIGATION
# FEATURE LOGIC
result = None
title = None
tab1, tab_trending, tab2, tab3 = st.tabs([
    "🎧 AI Generate",
    "🌍 Trending Top 50",
    "📊 AI Clustering",
    "🎯 AI Recommendation"
])

with tab1:
    st.subheader("🎧 AI Music Generation")

    sub_feature = st.selectbox(
        "Select Generation Type",
        [
            "🎧 Generate by Mood",
            "🎼 Generate by Genre",
            "🔥 Remix Song",
            "✍ Generate Full Lyrics",
            "🎵 Auto Song Title"
        ]
    )

    # 🎧 MOOD
    if sub_feature == "🎧 Generate by Mood":
        mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
        if st.button("Generate Music Idea 🎶", key="mood_btn"):
            loading_animation()
            if collab_artist:
                prompt = f"You are a {producer_mode}. Create a {mood} mood music concept in collaboration with {collab_artist}."
            else:
                prompt = f"You are a {producer_mode}. Create a {mood} mood music concept."
            result = generate_with_retry(prompt, temperature)
            title = f"{mood} Mood Concept"
            st.success("✨ Generated Music Concept")
            st.write(result)
            song_structure_visualizer()

    # 🎼 GENRE
    elif sub_feature == "🎼 Generate by Genre":
        genre = st.selectbox("Select Genre", ["Pop", "Classical", "Lo-fi", "EDM"])
        if st.button("Generate Genre Music 🎼", key="genre_btn"):
            loading_animation()
            if collab_artist:
                prompt = f"You are a {producer_mode}. Create a {genre} music concept in collaboration with {collab_artist}."
            else:
                prompt = f"You are a {producer_mode}. Create a {genre} music concept."
            result = generate_with_retry(prompt, temperature)
            title = f"{genre} Genre Concept"
            st.success("🎶 Generated Genre Concept")
            st.write(result)
            song_structure_visualizer()

    # 🔥 REMIX
    elif sub_feature == "🔥 Remix Song":
        text = st.text_area("Enter Song Description")
        if st.button("Remix with AI 🔥", key="remix_btn"):
            loading_animation()
            if collab_artist:
                prompt = f"You are a {producer_mode}. Remix this song creatively in collaboration with {collab_artist}:\n{text}"
            else:
                prompt = f"You are a {producer_mode}. Remix this song creatively:\n{text}"
            result = generate_with_retry(prompt, temperature)
            title = "Remixed Song Version"
            st.success("🔥 AI Remixed Version")
            st.write(result)

    # ✍ LYRICS
    elif sub_feature == "✍ Generate Full Lyrics":
        mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
        genre = st.selectbox("Select Genre", ["Pop", "Classical", "Lo-fi", "EDM"])
        if st.button("Generate Complete Song 🎤", key="lyrics_btn"):
            loading_animation()
            if collab_artist:
                prompt = f"Write full song lyrics in {genre} style with {mood} mood in collaboration with {collab_artist}."
            else:
                prompt = f"Write full song lyrics in {genre} style with {mood} mood."
            result = generate_with_retry(prompt, temperature)
            title = "Full Song Lyrics"
            st.success("🎵 Full Song Lyrics")
            st.write(result)

    # 🎵 TITLE
    elif sub_feature == "🎵 Auto Song Title":
        mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
        genre = st.selectbox("Select Genre", ["Pop", "Classical", "Lo-fi", "EDM"])
        if st.button("Generate Song Titles 🎶", key="title_btn"):
            loading_animation()
            if collab_artist:
                prompt = f"Generate 5 catchy titles for a {genre} song with {mood} mood in collaboration with {collab_artist}."
            else:
                prompt = f"Generate 5 catchy titles for a {genre} song with {mood} mood."
            result = generate_with_retry(prompt, temperature)
            title = "Generated Song Titles"
            st.success("🎼 Song Titles")
            st.write(result)

with tab_trending:

    st.subheader("🌍 Top 50 World Trending Songs")

    trending_songs = [
        "1. DtMF - Bad Bunny",
        "2. BAILE INoLVIDABLE - Bad Bunny",
        "3. NUEVAYoL - Bad Bunny",
        "4. EoO - Bad Bunny",
        "5. Tití Me Preguntó - Bad Bunny",
        "6. VOY A LLeVARTE PA PR - Bad Bunny",
        "7. End of Beginning - Djo",
        "8. LA CANCIÓN - J Balvin ft. Bad Bunny",
        "9. The Fate of Ophelia - Taylor Swift",
        "10. Man I Need - Olivia Dean",
        "11. back to friends - sombr",
        "12. Qué Pasaría... - Rauw Alejandro ft. Bad Bunny",
        "13. SOMETHING WEIRD - UNKNOWN",   # placeholder
        "14. Ordinary - Alex Warren",
        "15. So Easy (To Fall In Love) - Olivia Dean",
        "16. WILDFLOWER - Billie Eilish",
        "17. BIRDS OF A FEATHER - Billie Eilish",
        "18. I Just Might - Bruno Mars",
        "19. Raindance (feat. Tems) - Dave",
        "20. Lush Life - Zara Larsson",
        "21. Golden (w/ Ejae, Audrey Nuna & more) - Various Artists",
        "22. WHERE IS MY HUSBAND! - RAYE",
        "23. Don’t Say You Love Me - Jin",
        "24. Golden Hour - JVKE",
        "25. BAD HABITS - Ed Sheeran",
        "26. As It Was - Harry Styles",
        "27. Calm Down - Rema & Selena Gomez",
        "28. Flowers - Miley Cyrus",
        "29. Levitating - Dua Lipa",
        "30. Stay - The Kid LAROI & Justin Bieber",
        "31. Shape of You - Ed Sheeran",
        "32. Anti-Hero - Taylor Swift",
        "33. Kill Bill - SZA",
        "34. Unholy - Sam Smith & Kim Petras",
        "35. Sunroof - Nicky Youre & dazy",
        "36. Rockstar - Post Malone ft. 21 Savage",
        "37. Heat Waves - Glass Animals",
        "38. Someone You Loved - Lewis Capaldi",
        "39. Believer - Imagine Dragons",
        "40. Perfect - Ed Sheeran",
        "41. Uptown Funk - Mark Ronson ft. Bruno Mars",
        "42. Blinding Lights - The Weeknd",
        "43. Old Town Road - Lil Nas X",
        "44. Cheap Thrills - Sia ft. Sean Paul",
        "45. Havana - Camila Cabello ft. Young Thug",
        "46. Water - Tyla",
        "47. Easy On Me - Adele",
        "48. Peaches - Justin Bieber ft. Giveon, Daniel Caesar",
        "49. Counting Stars - OneRepublic",
        "50. Take You Dancing - Jason Derulo"
    ]

    for song in trending_songs:
        st.markdown(f"🎵 {song}")


with tab2:

    st.subheader("📊 K-Means Song Clustering")

    clustered_df = train_kmeans()
    cluster_groups = clustered_df.groupby("cluster")["track_name"].apply(list)

    for cluster, songs in cluster_groups.items():
        st.markdown(f"### 🎵 Cluster {cluster}")
        for song in songs[:10]:
            st.write(f"- {song}")

    st.success("Clustering Completed using K-Means Algorithm")

with tab3:

    st.subheader("🎯 ML-Based Song Recommendation")

    selected_song = st.selectbox("Select a Song", df["track_name"].unique())

    if st.button("Recommend Similar Songs 🎵", key="rec_btn"):
        recommendations = recommend_songs_by_features(selected_song)

        st.success("Top Recommended Songs:")
        for _, row in recommendations.iterrows():
            st.write(f"🎵 {row['track_name']}  —  {row['artists']}")


# Save Generated Output
if result and title:
    save_output(title, result)



# HISTORY SECTION
st.markdown("---")

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("### 🕘 Saved History")
with col2:
    if st.button("🗑 Clear"):
        st.session_state.history = []
        save_history()
        st.success("History Cleared!")

if not st.session_state.history:
    st.info("No history available yet.")
else:
    for index, item in enumerate(st.session_state.history):
        with st.expander(f"🎵 {item['title']} | {item['timestamp']}"):
            st.write(item["content"])


# Footer
st.markdown("""
<hr>
<div class="footer">
    Made with ❤️ using Generative AI & Machine Learning <br>
    © 2026 HarmonyAI | Created by Samruddhi Mahale
</div>
""", unsafe_allow_html=True)