# src/scripts/recommender.py

# THIS A FILE FUNCTIONS IS CALLED BY "model.py" 
# THIS FILE CONTAINS FUNCTIONS REQUIRED BY model.py



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import pickle
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, model_dir='src/trained_model'):
        self.model_dir = Path(model_dir)
        self.processed_data_path = self.model_dir/'processed_data.pkl'
        self.similarity_matrix_path = self.model_dir/'similarity_matrix.pkl'
        self.vectorizer_path = self.model_dir/'vectorizer.pkl'
        self.feature_names_path = self.model_dir/'feature_names.pkl'
        self.df = None
        self.similarity_matrix = None
        self.vectorizer = None

# to create and save trained model pickle files
    def create_similarity_matrix(self,df):
        """Create and save similarity matrix from processed data."""
        try:
            logger.info("Creating similarity matrix...")
            
            # Create vectors
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(df['tags']).toarray()
            similarity_matrix = cosine_similarity(vectors)

            # Create models directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Save all artifacts
            logger.info("Saving model artifacts...")
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(cv, f)

            with open(self.similarity_matrix_path, 'wb') as f:
                pickle.dump(similarity_matrix, f)

            df.to_pickle(self.processed_data_path)

            feature_names = cv.get_feature_names_out()
            with open(self.feature_names_path, 'wb') as f:
                pickle.dump(feature_names, f)

            logger.info("Successfully created and saved all model artifacts")
            self.df = df
            self.similarity_matrix = similarity_matrix
            self.vectorizer = cv
            
            return similarity_matrix

        except Exception as e:
            logger.error(f"Error in create_similarity_matrix: {str(e)}")
            raise

# CALLED BY "model.py" TO READ PICKLE FILES FOR PREDICTION
    def load_model_artifacts(self):
        """Load saved model artifacts."""
        try:
            logger.info("Loading model artifacts...")
            self.df = pd.read_pickle(self.processed_data_path)
            with open(self.similarity_matrix_path, 'rb') as f:
                self.similarity_matrix = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Model artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            raise



# this will be called in recommend_movies 👇method(below fuction)
    def find_closest_title(self, input_title, score_cutoff=60):
        """Find the closest matching movie title using fuzzy string matching."""
        input_title = input_title.lower()
        title_list = self.df['title'].tolist()
        matches = process.extractBests(
            input_title,
            title_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score_cutoff,
            limit=5
        )
        return matches[0][0] if matches else None
    def recommend_movies(self, movie_title, n_recommendations=5):
        """Get movie recommendations based on similarity with fuzzy matching."""
        try:
            if self.df is None or self.similarity_matrix is None:
                raise ValueError("Model artifacts are not loaded. Please load them first.")

            # Find closest matching title
            matched_title = self.find_closest_title(movie_title)

            if matched_title is None:
                logger.warning(f"No close matches found for '{movie_title}'")
                return [], None

            # Get movie index
            movie_index = self.df[self.df['title'] == matched_title].index[0]
            distances = self.similarity_matrix[movie_index]

            # Get similar movies
            movie_list = sorted(list(enumerate(distances)), 
                              reverse=True, 
                              key=lambda x: x[1])[1:n_recommendations+1]

            # Get recommendations with similarity scores
            recommendations = [
                {
                    'title': self.df.iloc[i[0]].title,
                    'similarity': round(i[1] * 100, 2)
                }
                for i in movie_list
            ]

            return recommendations, matched_title

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [], None


if __name__ == "__main__":
    pass
