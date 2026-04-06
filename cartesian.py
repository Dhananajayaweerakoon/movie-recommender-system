import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading model data...")
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
new_df = pd.DataFrame(movie_dict)

# ---------------------------------------------------------
# CHOOSE THE TWO MOVIES TO PLOT
# ---------------------------------------------------------
movie1 = "Sleep Dealer"
movie2 = "The Matrix Reloaded" 

print(f"Calculating vector angles for '{movie1}' and '{movie2}'...")

try:
    # THE FIX: This gets the exact 0-based list position so it 
    # perfectly matches the mathematical similarity matrix.
    idx1 = new_df['title'].tolist().index(movie1)
    idx2 = new_df['title'].tolist().index(movie2)
except ValueError:
    print("Error: Movie not found. Check spelling!")
    exit()

# 1. Get the exact similarity score calculated by your ML model
sim_score = similarity[idx1][idx2]

# Ensure it doesn't break floating point rules (cap at 1.0)
sim_score = min(sim_score, 1.0)

# 2. Convert the similarity score back into an Angle
# sim_score = cos(theta) ---> theta = arccos(sim_score)
angle_rad = np.arccos(sim_score)
angle_deg = np.degrees(angle_rad)

# 3. Create Vector Coordinates for the Cartesian Plane
# Let Movie 1 sit flat on the X-axis (Length 1, Angle 0)
x1, y1 = 1.0, 0.0

# Let Movie 2 sit at the calculated angle (Using Trigonometry: x=cos, y=sin)
x2, y2 = np.cos(angle_rad), np.sin(angle_rad)

# ---------------------------------------------------------
# DRAWING THE CARTESIAN GRAPH
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
ax = plt.gca()

# Draw the arrows (Vectors)
ax.quiver(0, 0, x1, y1, angles='xy', scale_units='xy', scale=1, color='#e94560', width=0.015, label=movie1)
ax.quiver(0, 0, x2, y2, angles='xy', scale_units='xy', scale=1, color='#0f3460', width=0.015, label=movie2)

# Draw an arc to show the angle between them
from matplotlib.patches import Arc
arc = Arc((0,0), 0.5, 0.5, angle=0, theta1=0, theta2=angle_deg, color='green', linewidth=2)
ax.add_patch(arc)

# Add Text Labels
plt.text(x1 + 0.05, y1, movie1, fontsize=12, fontweight='bold', color='#e94560', va='center')
plt.text(x2 + 0.05, y2 + 0.05, movie2, fontsize=12, fontweight='bold', color='#0f3460')
plt.text(0.2, 0.1, f"{angle_deg:.1f}°\n(Similarity: {sim_score:.2f})", fontsize=11, color='green', fontweight='bold')

# Setup the Cartesian Grid
plt.xlim(-0.2, 1.5)
plt.ylim(-0.2, 1.5)
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)

plt.title(f"Machine Learning Cartesian Vector Space", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Vector Dimension X", fontsize=12)
plt.ylabel("Vector Dimension Y", fontsize=12)

# Save the picture
filename = "cartesian_vectors.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Success! Saved as {filename}")