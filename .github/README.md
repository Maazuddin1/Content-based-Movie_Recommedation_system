# Movie Recommendation System

A content-based movie recommendation system built with Python, scikit-learn, and Flask.

## Features
- Content-based movie recommendations
- Web interface with autocomplete search
- RESTful API endpoints
- Comprehensive error handling
- Logging system
- Unit tests

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Place your movie datasets in the `data` folder
2. Run the preprocessing pipeline:
```bash
python src/app/main.py
```
3. Start the Flask application:
```bash
python src/app/app.py
```

## Testing
```bash
pytest tests/
```

## License
MIT License