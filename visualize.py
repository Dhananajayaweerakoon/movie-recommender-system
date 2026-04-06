import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading model data...")
# Load your saved "brain"
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
new_df = pd.DataFrame(movie_dict)

# The movie you want to feature in your LinkedIn post
target_movie = "The Matrix" 
print(f"Generating heatmap for: {target_movie}")

# 1. Find the index of the target movie
movie_idx = new_df[new_df['title'] == target_movie].index[0]

# 2. Get the top 5 most similar movies (plus the target movie itself = 6 movies total)
distances = similarity[movie_idx]
movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[0:6]

# Extract their specific indices and titles
indices = [i[0] for i in movies_list]
titles = [new_df.iloc[i].title for i in indices]

# 3. Create a small 6x6 matrix just for these specific movies
matrix_subset = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        # Grab the exact math your model calculated between these movies
        matrix_subset[i][j] = similarity[indices[i]][indices[j]]

# 4. Draw the Heatmap
plt.figure(figsize=(10, 8)) # Size of the image
# cmap="mako" gives a very professional, high-tech dark blue/green look
sns.heatmap(matrix_subset, annot=True, xticklabels=titles, yticklabels=titles, cmap="mako", fmt=".2f", linewidths=.5)

# Add titles and labels
plt.title(f"Cosine Similarity Matrix: {target_movie}", pad=20, fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

# 5. Save the picture
filename = "linkedin_heatmap.png"
plt.savefig(filename, dpi=300) # dpi=300 makes it high-resolution
print(f"Success! Image saved as '{filename}'")