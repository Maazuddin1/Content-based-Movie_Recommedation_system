import sys
import logging
import sys
from pathlib import Path
# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# Import from scripts
from src.scripts.preprocessing import DataPreprocessor
from src.scripts.recommender import MovieRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Define base project path
        base_path = Path(__file__).resolve().parents[2]

        # File paths
        credits_path = base_path/'data'/'credits.csv'
        movies_path = base_path/'data'/'movies.csv'
        processed_path = base_path/'data'/'processed_dataset'/'movies_preprocessed.csv'
        model_dir = base_path/'src'/'trained_model'

        # Ensure necessary directories exist
        model_dir.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)


        # Data Preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor(
                            credits_path=credits_path,
                            movies_path=movies_path,
                            output_path=processed_path
        )
        preprocessor.load_data()
        processed_df = preprocessor.preprocess_data()
        preprocessor.save_preprocessed_data()
        logger.info("Data preprocessing completed successfully.")



        # Model Training
        logger.info("Starting model training...")
        recommender = MovieRecommender(model_dir=str(model_dir))
        recommender.create_similarity_matrix(processed_df)
        logger.info("Model training completed successfully.")
        # Model Testing with a Sample Recommendation
        logger.info("Testing model with a sample recommendation...")
        recommender.load_model_artifacts()
        movie_title = "Dark Knight"
        recommendations, matched_title = recommender.recommend_movies(movie_title)



        # Display Recommendations
        if recommendations and matched_title:
            print(f"\nRecommendations for '{matched_title}':")
            for movie in recommendations:
                print(f"- {movie['title']} ({movie['similarity']}% match)")
        else:
            print(f"No recommendations found for '{movie_title}'.")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    # Add project root to sys.path for module resolution
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    main()