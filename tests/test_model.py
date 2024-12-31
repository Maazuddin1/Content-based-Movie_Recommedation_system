import pytest
from src.scripts.recommender import MovieRecommender
import pandas as pd
import numpy as np

@pytest.fixture
def sample_movies_df():
    return pd.DataFrame({
        'movie_id': [1, 2, 3],
        'title': ['The Dark Knight', 'Inception', 'Interstellar'],
        'tags': [
            'batman dark knight action',
            'dreams inception thriller',
            'space interstellar scifi'
        ]
    })

def test_movie_recommender_initialization():
    recommender = MovieRecommender()
    assert recommender is not None
    assert recommender.df is None
    assert recommender.similarity_matrix is None

def test_find_closest_title(sample_movies_df):
    recommender = MovieRecommender()
    recommender.df = sample_movies_df
    
    # Test exact match
    assert recommender.find_closest_title('The Dark Knight') == 'The Dark Knight'
    
    # Test partial match
    assert recommender.find_closest_title('Dark Knight') == 'The Dark Knight'
    
    # Test no match
    assert recommender.find_closest_title('Nonexistent Movie') is None

def test_recommend_movies(sample_movies_df):
    recommender = MovieRecommender()
    recommender.df = sample_movies_df
    
    # Create a simple similarity matrix for testing
    recommender.similarity_matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Test recommendations
    recommendations, matched_title = recommender.recommend_movies('The Dark Knight')
    
    assert matched_title == 'The Dark Knight'
    assert len(recommendations) == 2  # Should return 2 recommendations
    assert recommendations[0]['title'] in ['Inception', 'Interstellar']
    assert 0 <= recommendations[0]['similarity'] <= 100

if __name__ == '__main__':
    pytest.main(['-v'])
