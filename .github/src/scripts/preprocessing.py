# src/scripts/preprocessing.py
import pandas as pd
import numpy as np
import ast
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, credits_path,movies_path,output_path):
        self.credits_path = Path(credits_path)
        self.movies_path = Path(movies_path)
        self.output_path = Path(output_path)
        self.data = None

    def load_data(self):
        """Load and merge the movies and credits datasets."""
        try:
            logger.info("Loading datasets...")
            credits_df = pd.read_csv(self.credits_path)
            movies_df = pd.read_csv(self.movies_path)
            self.data = movies_df.merge(credits_df)
            logger.info(f"Successfully loaded {len(self.data)} movies")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def convert_literals(obj):
        """Convert string literals to list of names."""
        return [item['name'] for item in ast.literal_eval(obj)]

    @staticmethod
    def cast3only(obj):
        """Extract top 3 cast members."""
        cast = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                cast.append(i['name'])
                counter += 1
        return cast

    @staticmethod
    def find_director(obj):
        """Extract director name from crew."""
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    def preprocess_data(self):
        """Main preprocessing function."""
        try:
            if self.data is None:
                self.load_data()
                
            logger.info("Preprocessing data...")
            # Select required columns
            self.data = self.data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            
            # Drop missing values
            self.data = self.data.dropna()
            
            # Apply transformations
            self.data['genres'] = self.data['genres'].apply(self.convert_literals)
            self.data['keywords'] = self.data['keywords'].apply(self.convert_literals)
            self.data['cast'] = self.data['cast'].apply(self.cast3only)
            self.data['crew'] = self.data['crew'].apply(self.find_director)
            
            # Process text data
            self.data['overview'] = self.data['overview'].apply(lambda x: x.split())
            
            # Remove spaces from all lists
            for col in ['overview', 'genres', 'keywords', 'cast']:
                self.data[col] = self.data[col].apply(lambda x: [i.replace(' ', '') for i in x])
            
            # Combine features
            self.data['tags'] = self.data['overview'] + self.data['genres'] + self.data['keywords'] + self.data['cast'] + self.data['crew']
            
            self.data = self.data[['movie_id', 'title', 'tags']]
            self.data['tags'] = self.data['tags'].apply(lambda x: ' '.join(x))
            self.data['tags'] = self.data['tags'].apply(lambda x: x.lower())

            logger.info("Data preprocessing completed.")
            return self.data

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def save_preprocessed_data(self):
        """Save preprocessed data to a csv file."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(self.output_path, index=False)
            logger.info(f"Saved preprocessed data to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise

def main():
    # Set file paths
    credits_path = 'data/credits.csv'
    movies_path = 'data/movies.csv'
    output_path = 'data/processed_dataset/movies_preprocessed.csv'
    
    # Create an instance of DataPreprocessor
    preprocessor = DataPreprocessor(credits_path, movies_path, output_path)
    
    try:
        # Load and preprocess data
        preprocessor.load_data()
        preprocessor.preprocess_data()
        # Save preprocessed data
        preprocessor.save_preprocessed_data()
        
        logger.info("Preprocessing pipeline executed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the preprocessing pipeline: {str(e)}")


if __name__ == '__main__':
    main()