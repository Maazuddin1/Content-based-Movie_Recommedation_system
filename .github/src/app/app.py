from flask import Flask, render_template, request
import sys
from pathlib import Path
# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.scripts.recommender import MovieRecommender
# app.py (Flask backend)
app = Flask(__name__,template_folder='../templates')

recommender = MovieRecommender(model_dir='src/trained_model')

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    matched_title = ""
    
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommender.load_model_artifacts()
        recommendations, matched_title = recommender.recommend_movies(movie_title)
        
    return render_template('index.html', recommendations=recommendations, matched_title=matched_title)

if __name__ == "__main__":
    app.run(debug=True)

