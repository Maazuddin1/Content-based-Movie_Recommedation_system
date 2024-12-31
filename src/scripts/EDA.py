# src/scripts/EDA.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def perform_eda(credits_df, movies_df):
    """Perform exploratory data analysis on the datasets."""
    print("Dataset Information:")
    print("\nMovies Dataset Shape:", movies_df.shape)
    print("Credits Dataset Shape:", credits_df.shape)


    pd.options.display.float_format = '{:.2f}'.format
    

    # Missing values analysis
    print("\nMissing Values:")
    print(movies_df.isnull().sum())
    
    # Basic statistics
    print("\nNumerical Columns Statistics:")
    print(movies_df.describe())
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    movies_df['vote_average'].hist()
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    #plt.savefig('data/visualizations/ratings_dist.png')
    #plt.close()


def main():
    # Load the preprocessed data
    credits_df = pd.read_csv('g:/my projects/Content-based-Movie_Recommedation_system/data/credits.csv')
    movies_df = pd.read_csv('g:/my projects/Content-based-Movie_Recommedation_system/data/movies.csv')
    
    # Perform EDA
    perform_eda(credits_df, movies_df)

if __name__ == "__main__":
    main()