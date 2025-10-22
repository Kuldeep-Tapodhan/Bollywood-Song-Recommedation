from flask import Flask, render_template, jsonify, request
import pickle as pk
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the dataset and features
with open("Song_list.pkl", "rb") as f:
    song_df = pk.load(f)

# Load the feature matrix (X)
with open("feature_matrix.pkl", "rb") as f:
    X = pk.load(f)

# Initialize K-Nearest Neighbors
knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 neighbors: input song + 5 recommendations
knn.fit(X)

# Define the recommendation function
def recommend(song):
    try:
        song_index = song_df[song_df['Song-Name'] == song].index[0]
    except IndexError:
        return []  # Return empty list if song not found

    song_features = X[song_index]  # Get the features of the selected song

    # Find nearest neighbors (excluding the input song itself)
    distances, indices = knn.kneighbors(song_features, n_neighbors=6)

    # Limit to top 5 recommendations (excluding the input song itself)
    recommended_songs = [
        song_df.iloc[i]['Song-Name']
        for i in indices.flatten() if i != song_index
    ][:5]

    return recommended_songs

# Route for serving index.html
@app.route('/')
def index():
    song_names = song_df['Song-Name'].tolist()
    return render_template('index.html', song_names=song_names)

# API route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend_api():
    selected_song = request.json.get('selected_song')
    recommended_songs = recommend(selected_song)
    return jsonify(recommended_songs)

