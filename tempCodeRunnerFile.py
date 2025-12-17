knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 neighbors: input song + 5 recommendations
knn.fit(X)