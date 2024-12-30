# src/scripts/train_model.py
from pathlib import Path
from preprocessing import DataPreprocessor
import sys
import os
from recommender import MovieRecommender
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get absolute paths
        base_path = Path(__file__).resolve().parents[2]
        
        # Setup paths
        credits_path = base_path/'data'/'credits.csv'
        movies_path = base_path/'data'/'movies.csv'
        processed_path = base_path/'data'/'processed_dataset'/'movies_preprocessed.csv'
        model_dir = base_path/'src'/'trained_model'

        logger.info("Preprocessed data found. Loading it directly...")
        processed_df = pd.read_csv(processed_path)  # Adjust based on file format (e.g., CSV, pickle)
        logger.info("Starting model training...")
        
        # Initialize and train recommender
        recommender = MovieRecommender(model_dir=str(model_dir))
        recommender.create_similarity_matrix(processed_df)
        logger.info("Model training completed successfully!")
        
        # Test the model
        logger.info("Testing model with a sample recommendation...")
        recommender.load_model_artifacts()
        recommendations, matched_title = recommender.recommend_movies("The Dark Knight")
        
        if recommendations and matched_title:
            print(f"\nRecommendations for {matched_title}:")
            for movie in recommendations:
                print(f"- {movie['title']} ({movie['similarity']}% match)")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()