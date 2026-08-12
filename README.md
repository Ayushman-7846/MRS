# Movie Recommendation System (NLP)

A Natural Language Processing (NLP) based **Movie Recommendation
System** built in Python and developed in Google Colab. The project uses
movie metadata from the TMDB 5000 dataset, semantic embeddings from
Sentence Transformers, and cosine similarity to generate movie
recommendations and semantic search results.

## Project Overview

This project combines traditional movie metadata processing with modern
semantic text embeddings.

The notebook:

-   Loads and merges the TMDB 5000 movies and credits datasets.
-   Extracts useful information from genres, keywords, cast, crew, and
    movie overviews.
-   Creates a combined textual representation called `tags`.
-   Generates semantic embeddings using `all-MiniLM-L6-v2`.
-   Calculates movie-to-movie similarity using cosine similarity.
-   Provides multiple recommendation modes:
    -   Movie-title based recommendation
    -   Natural-language semantic search
    -   Mood-based recommendation
    -   Smart recommendation that automatically detects whether the
        input is a movie title or a natural-language query
-   Performs sentiment analysis using TextBlob.
-   Generates several visualizations for exploring popularity, ratings,
    genres, similarity, recommendation scores, and sentiment.

## Features

### 1. Movie-Based Recommendation

Provide an existing movie title and receive the most similar movies.

``` python
recommend("Avatar")
```

The system compares the semantic representation of the selected movie
with all other movies and returns the top recommendations.

### 2. Semantic Search

The system also accepts natural-language descriptions rather than
requiring an exact movie title.

``` python
semantic_search("dark psychological thriller")
```

The query is converted into an embedding and compared against the movie
embeddings.

### 3. Mood-Based Recommendation

A predefined mood-to-keyword mapping converts moods into semantic movie
descriptions.

``` python
recommend_by_mood("happy")
```

Current mappings include:

  Mood       Semantic Query
  ---------- -------------------------
  happy      comedy fun friendship
  sad        emotional drama
  thriller   crime suspense mystery
  scifi      space future technology

Other inputs can also be passed to the mood recommender and are used as
semantic-search queries when no predefined mapping exists.

### 4. Smart Recommender

The smart recommender determines whether the user's input exactly
matches a movie title.

``` python
smart_recommender("I want a sci-fi movie")
```

-   If the input is a movie title, movie-based recommendation is used.
-   Otherwise, semantic search is used.

### 5. Sentiment Analysis

The notebook uses TextBlob to calculate sentiment polarity.

``` python
analyze_sentiment("This movie was amazing and inspiring")
```

The current implementation returns:

-   `POSITIVE` when polarity is greater than `0`
-   `NEGATIVE` otherwise

Therefore, neutral sentiment is currently classified as `NEGATIVE`.

## Technology Stack

-   **Python**
-   **Google Colab**
-   **Pandas** --- data loading and manipulation
-   **NumPy** --- numerical operations
-   **NLTK** --- stopwords, stemming, and NLP utilities
-   **TextBlob** --- sentiment analysis
-   **Scikit-learn** --- cosine similarity
-   **Sentence Transformers** --- semantic embeddings
-   **Matplotlib** --- data visualization

## Dataset

The notebook expects two TMDB 5000 dataset files:

``` text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The files are uploaded through Google Colab:

``` python
from google.colab import files
uploaded = files.upload()
```

The two datasets are merged using the movie title.

The project keeps the following movie information:

``` text
movie_id
title
overview
genres
keywords
cast
crew
vote_average
popularity
```

## NLP / Data Processing Pipeline

The overall pipeline is:

``` text
TMDB Movies Dataset
        +
TMDB Credits Dataset
        ↓
      Merge
        ↓
   Select Columns
        ↓
    Remove NaN
        ↓
 Parse Genres / Keywords
        ↓
   Extract Top 5 Cast
        ↓
    Extract Director
        ↓
   Process Overview
        ↓
 Combine Information
        ↓
       Tags
        ↓
 Sentence Transformer
(all-MiniLM-L6-v2)
        ↓
 Movie Embeddings
        ↓
 Cosine Similarity
        ↓
 Recommendations
```

### Tag Creation

The project combines:

-   Movie overview
-   Genres
-   Keywords
-   Top 5 cast members
-   Director

These values are combined into a single `tags` field.

The tags are converted into lowercase text before being passed to the
Sentence Transformer model.

## Semantic Embeddings

The project uses:

``` python
SentenceTransformer('all-MiniLM-L6-v2')
```

Each movie's tags are converted into a numerical embedding.

The embeddings are then compared using cosine similarity:

``` python
similarity = cosine_similarity(embeddings)
```

This creates a similarity matrix that is used for movie recommendations.

## Recommendation Logic

For a selected movie:

1.  Find the movie in the dataset.
2.  Retrieve its similarity scores.
3.  Sort movies by similarity.
4.  Remove the selected movie itself.
5.  Return the top `N` movie titles.

The default number of recommendations is 5:

``` python
recommend(movie, top_n=5)
```

## Visualizations

The notebook includes several exploratory and recommendation-related
visualizations.

### Movie and Dataset Analysis

-   Top 10 most popular movies
-   Distribution of movie ratings
-   Top 10 highest-rated movies
-   Top 10 most common genres

### Recommendation Analysis

-   Similarity scores for recommendations
-   Distribution of similarity scores
-   Top-N recommendation score curve
-   Diversity / recommendation-gap analysis
-   Distribution of average similarity scores
-   Sentiment score vs. average recommendation similarity

These visualizations help inspect both the dataset and the behavior of
the recommendation model.

## Example Usage

### Recommend movies similar to a movie

``` python
recommend("Avatar")
```

### Search using a description

``` python
semantic_search("dark psychological thriller")
```

### Recommend by mood

``` python
recommend_by_mood("happy")
```

### Use the smart recommender

``` python
smart_recommender("I want a sci-fi movie")
```

or:

``` python
smart_recommender("I want a thriller movie")
```

### Analyze sentiment

``` python
analyze_sentiment("This was an excellent movie")
```

## Installation

The notebook installs the required packages with:

``` bash
pip install nltk textblob scikit-learn sentence-transformers
```

NLTK resources are also downloaded:

``` python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Running the Project

### Option 1 --- Google Colab

1.  Open the notebook in Google Colab.
2.  Install the required dependencies.
3.  Run the NLTK downloads.
4.  Upload:
    -   `tmdb_5000_movies.csv`
    -   `tmdb_5000_credits.csv`
5.  Run the cells sequentially.
6.  Test the recommendation functions.

### Option 2 --- Local Jupyter Environment

Install the required Python packages:

``` bash
pip install numpy pandas nltk textblob scikit-learn sentence-transformers matplotlib
```

Then open:

``` text
MRS_nlp_project.ipynb
```

Make sure both TMDB CSV files are available in the notebook's working
directory.

## Project Structure

``` text
MRS/
│
├── MRS_nlp_project.ipynb
├── README.md
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
```

The exact repository structure may vary depending on how the project is
organized.

## Main Functions

  -----------------------------------------------------------------------
  Function                            Purpose
  ----------------------------------- -----------------------------------
  `convert()`                         Extracts names from structured
                                      movie metadata

  `convert_cast()`                    Extracts the first five cast
                                      members

  `fetch_director()`                  Extracts the movie director

  `collapse()`                        Removes spaces from metadata names

  `recommend()`                       Recommends movies similar to a
                                      given movie

  `semantic_search()`                 Searches movies using a
                                      natural-language query

  `recommend_by_mood()`               Generates recommendations based on
                                      mood

  `analyze_sentiment()`               Performs TextBlob sentiment
                                      analysis

  `smart_recommender()`               Chooses title-based recommendation
                                      or semantic search
  -----------------------------------------------------------------------

## Model

### Sentence Transformer

Model:

``` text
all-MiniLM-L6-v2
```

It is used to generate semantic representations of movie tags and user
queries.

### Similarity Metric

The project uses:

``` text
Cosine Similarity
```

to measure semantic similarity between movie embeddings and between a
query embedding and movie embeddings.

## Limitations

The current notebook has several limitations:

-   Recommendations depend on the available TMDB metadata.
-   The system is primarily content-based and does not use user ratings
    or collaborative filtering.
-   Movie-title matching requires an exact title match after
    lowercasing.
-   The mood system uses a manually defined mood-to-query mapping.
-   The complete similarity matrix is computed in memory.
-   Sentiment analysis is performed on movie tags rather than user
    reviews.
-   The current sentiment function treats zero/neutral polarity as
    `NEGATIVE`.
-   There is no persistent web application or database layer in the
    notebook itself.
-   Recommendation quality depends on the semantic representation
    generated from the selected metadata.

## Future Improvements

Possible extensions include:

-   Add fuzzy movie-title matching.
-   Add movie posters and metadata through an external movie API.
-   Add collaborative filtering using user ratings.
-   Build a hybrid recommendation system.
-   Improve mood detection using a dedicated NLP classifier.
-   Use movie reviews for sentiment analysis.
-   Add user profiles and personalized recommendations.
-   Add recommendation explanations.
-   Optimize similarity search using FAISS or another vector-search
    system.
-   Deploy the recommender as a web application or API.
-   Add evaluation metrics such as Precision@K, Recall@K, and NDCG.

## Author

**Ayushman Panda**

GitHub: [Ayushman-7846](https://github.com/Ayushman-7846)

## Project Notebook

The project is implemented in the accompanying Jupyter/Google Colab
notebook:

``` text
MRS_nlp_project.ipynb
```

------------------------------------------------------------------------

**Note:** This README describes the implementation present in the
provided notebook. It does not assume functionality that is not
implemented in the current project.
