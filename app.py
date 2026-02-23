from flask import Flask, render_template, jsonify, request
import pickle as pk
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# -----------------------------
# Load Data Safely
# -----------------------------
try:
    with open("Song_list.pkl", "rb") as f:
        song_df = pk.load(f)

    with open("feature_matrix.pkl", "rb") as f:
        X = pk.load(f)

except Exception as e:
    print("Error loading pickle files:", e)
    song_df = pd.DataFrame()
    X = None

# -----------------------------
# Initialize KNN Model
# -----------------------------
if X is not None:
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(X)


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(song):
    if song_df.empty or X is None:
        return []

    try:
        song_index = song_df[song_df['Song-Name'] == song].index[0]
    except IndexError:
        return []

    song_features = X[song_index]

    distances, indices = knn.kneighbors(song_features.reshape(1, -1), n_neighbors=6)

    recommended_songs = [
        song_df.iloc[i]['Song-Name']
        for i in indices.flatten()
        if i != song_index
    ][:5]

    return recommended_songs


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    if song_df.empty:
        return "Data not loaded properly."

    song_names = song_df['Song-Name'].tolist()
    return render_template('index.html', song_names=song_names)


@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()

    if not data or "selected_song" not in data:
        return jsonify({"error": "No song selected"}), 400

    selected_song = data["selected_song"]
    recommended_songs = recommend(selected_song)

    return jsonify(recommended_songs)


# -----------------------------
# Render Required PORT Binding
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
