import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
data = pd.read_csv("goodreads.csv")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fill NaN values with empty string
data['title'] = data['title'].fillna('')
data['authors'] = data['authors'].fillna('')

# Construct TF-IDF matrix
tfidf_matrix_title = tfidf_vectorizer.fit_transform(data['title'])
tfidf_matrix_authors = tfidf_vectorizer.fit_transform(data['authors'])

# Compute cosine similarity matrix
cosine_sim_title = linear_kernel(tfidf_matrix_title, tfidf_matrix_title)
cosine_sim_authors = linear_kernel(tfidf_matrix_authors, tfidf_matrix_authors)


# Function to recommend books based on title and authors
def get_recommendations(book_title, author_name, cosine_sim_title, cosine_sim_authors, data):
    title_indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    author_indices = pd.Series(data.index, index=data['authors']).drop_duplicates()

    title_idx = title_indices[book_title]
    author_idx = author_indices[author_name]

    sim_scores_title = list(enumerate(cosine_sim_title[title_idx]))
    sim_scores_authors = list(enumerate(cosine_sim_authors[author_idx]))

    # Combine scores from title and authors
    sim_scores = [(i, (score_title + score_author) / 2) for i, (score_title, score_author) in
                  zip(sim_scores_title, sim_scores_authors)]

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return data['title'].iloc[book_indices]