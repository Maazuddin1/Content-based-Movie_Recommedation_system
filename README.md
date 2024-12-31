# Movie Recommendation System

## 🎥 Watch the Demo(click image below👇)

[![Watch on YouTube](https://www.youtube.com/watch?v=FWnURzv_P0Q)

## Overview
The **Movie Recommendation System** is a Python-based project designed to recommend movies to users based on content similarity. Using natural language processing and machine learning techniques, this system analyzes movie metadata to provide tailored recommendations.

---

## Features
1. **Data Preprocessing**: Cleans and merges raw movie datasets.
2. **Model Training**: Creates a similarity matrix using cosine similarity on vectorized features.
3. **Fuzzy Matching**: Matches user input to the closest movie title using fuzzy string matching.
4. **Recommendations**: Recommends movies with similarity scores.
5. **Modular Design**: Separate modules for preprocessing, training, and recommendations.

---

## Project Structure
```
├── data
│   ├── credits.csv               # Raw credits dataset
│   ├── movies.csv                # Raw movies dataset
│   └── processed_dataset
│       └── movies_processed.csv  # Preprocessed movie data
├── src
│   ├── scripts
│   │   ├── preprocessing.py      # Data preprocessing pipeline
│   │   ├── model.py              # Model training and artifact creation
│   │   ├── recommender.py        # Recommendation engine
│   └── trained_model
│       ├── similarity_matrix.pkl # Precomputed similarity matrix
│       ├── vectorizer.pkl        # Saved CountVectorizer instance
│       └── feature_names.pkl     # Feature names from vectorization
├── main.py                       # Central script for running pipelines
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```



---

## Technical Details
### Preprocessing
- Extracts key features (`genres`, `keywords`, `cast`, `crew`).
- Combines features into a unified text field (`tags`) for vectorization.

### Model
- **Vectorization**: Converts text into numerical vectors using `CountVectorizer`.
- **Similarity Matrix**: Computes cosine similarity between movie vectors.

### Recommendation Logic
1. Matches user input to the closest movie title using fuzzy string matching.
2. Fetches the most similar movies based on the similarity matrix.
3. Outputs a ranked list of recommendations with similarity scores.

---

## Example Output
```bash
Recommendations for 'The Dark Knight':
- The Dark Knight Rises (95.67% match)
- Batman Begins (89.23% match)
- Inception (82.45% match)
```

---

## Challenges and Solutions
- **Large Dataset**: Optimized vectorization using `CountVectorizer` with a feature limit.
- **Matching Errors**: Improved title matching accuracy with fuzzy string matching.

---

## Future Enhancements
1. Implement collaborative filtering for better recommendations.
2. Introduce a user-friendly web interface.
3. Integrate additional data sources, such as IMDb ratings and reviews.

---

## Contributing
Contributions are welcome! Fork the repository, make changes, and submit a pull request.

---

## Contact
For questions or suggestions, feel free to reach out:
- Email: [maazuddin173@gmail.com](mailto:maazuddin173@gmail.com)
- GitHub: [Maazuddin1](https://github.com/Maazuddin1)
