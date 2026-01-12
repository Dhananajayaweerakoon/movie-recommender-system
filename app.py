from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
import pandas as pd
import pickle

app = FastAPI()

# -------------------------------------------
# NEW: ALLOW FRONTEND TO TALK TO BACKEND (CORS)
# -------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# -------------------------------------------
# LOAD THE SAVED MODELS
# -------------------------------------------
print("Loading model files...")
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
new_df = pd.DataFrame(movie_dict)
print("Model loaded and ready!")

@app.get("/")
def read_root():
    return {"message": "Movie Recommender API is RUNNING!"}

@app.get("/recommend/{movie_title}")
def get_recommendations(movie_title: str):
    matches = new_df[new_df['title'].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
         raise HTTPException(status_code=404, detail="Movie not found")
    movie_idx = matches.index[0]
    distances = similarity[movie_idx]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
        
    return {"movie": matches.iloc[0]['title'], "recommendations": recommended_movies}