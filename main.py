import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # Changed from Tfidf
from sklearn.metrics.pairwise import cosine_similarity # Changed from linear_kernel
import ast # Needed to parse strings that look like lists like "[{'id':1, 'name':'Action'}]"
import pickle
# =========================================
# HELPER FUNCTIONS FOR DATA CLEANING
# =========================================

# Function 1: Extract standard lists (for Genres and Keywords)
# Turns "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]" into ['Action', 'Adventure']
def convert(obj):
    L = []
    # ast.literal_eval safely evaluates a string containing a Python literal
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Function 2: Extract top 3 actors
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

# Function 3: Fetch the director's name
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Function 4: Collapse spaces (e.g., "Brad Pitt" becomes "BradPitt")
# This is crucial so the model knows "Brad" and "Pitt" belong together.
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# ---------------------------------------------------------
# STEP 1: LOAD AND MERGE DATA
# ---------------------------------------------------------
print("Loading data files...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the two datasets based on movie title
movies = movies.merge(credits, on='title')

# Select only the columns we need for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Drop movies with missing info
movies.dropna(inplace=True)

print(f"Data merged! We have {movies.shape[0]} movies.")
print("-" * 30)

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (Cleaning the Messy Data)
# ---------------------------------------------------------
print("Cleaning data (Applying helper functions)...")

# 1. Apply the conversion functions to fix the messy JSON column formats
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# 2. Convert the overview string into a list of words
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# 3. Collapse spaces so names are treated as single tags
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# 4. CREATE THE "TAGS" SOUP
# Combine everything into one big list of tags for each movie
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new simpler dataframe with just ID, Title, and Tags
new_df = movies[['movie_id','title','tags']].copy()

# Convert the list of tags back into a single string string for the ML model
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) # lowercase everything

print("Data cleaned! Example 'soup' for first movie:")
print(new_df['tags'][0])
print("-" * 30)

# ---------------------------------------------------------
# STEP 3: VECTORIZATION & SIMILARITY
# ---------------------------------------------------------
print("Vectorizing tags and calculating similarity...")

# We switch to CountVectorizer here instead of TF-IDF.
# Why? Because with keywords/genres, repetition doesn't mean less importance.
# max_features=5000 means we only take the 5000 most frequent words to save memory.
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

print("Similarity Matrix calculated!")
print("-" * 30)

# ---------------------------------------------------------
# STEP 4: RECOMMENDATION FUNCTION
# ---------------------------------------------------------
def recommend(movie_title):
    # Check if movie exists in our database
    if movie_title not in new_df['title'].values:
         return ["Movie not found in database."]

    # Find index
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    
    # Get similarity scores
    distances = similarity[movie_index]
    
    # Sort and get top 5 (skipping the first one which is itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        # Add the movie title to the results list
        recommended_movies.append(new_df.iloc[i[0]].title)
        
    return recommended_movies

# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------
# Try "The Dark Knight Rises" again. The results should be better now!
# It should now prioritize Christopher Nolan movies or Christian Bale movies.
test_movie = "The Dark Knight Rises"
print(f"Recommendations for '{test_movie}' (using improved engine):")
results = recommend(test_movie)
for movie in results:
    print("-", movie)

print("\nTesting another one: 'Avatar'")
results = recommend("Avatar")
for movie in results:
    print("-", movie)
    # ... (all your previous code) ...

print("-" * 30)
print("Saving model files (Pickling)...")

# We need to save two things:

# 1. The dataframe (new_df) so we know the movie titles and IDs
# We convert it to a dictionary first as it's safer to pickle than a full dataframe often
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

# 2. The calculated similarity matrix
# This is the most important file! It holds the math.
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Files saved successfully!")
print("You should now see 'movie_dict.pkl' and 'similarity.pkl' in your folder.")