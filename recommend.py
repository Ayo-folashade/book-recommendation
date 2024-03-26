import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic, accuracy

# Load dataset using pandas
df = pd.read_csv('goodreads.csv')

# Define the reader
reader = Reader(rating_scale=(1, 5))

# Load dataset from DataFrame
data = Dataset.load_from_df(df[['isbn', 'title', 'rating_score']], reader)

# Split dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Collaborative Filtering based Recommendation System (User-Item based)
# Using KNN with cosine similarity
user_item_algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
user_item_algo.fit(trainset)
user_item_predictions = user_item_algo.test(testset)

# Collaborative Filtering based Recommendation System (Item-Item based)
# Using KNN with cosine similarity
item_item_algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
item_item_algo.fit(trainset)
item_item_predictions = item_item_algo.test(testset)

# Evaluate user-item based recommendations
user_item_rmse = accuracy.rmse(user_item_predictions)

# Evaluate item-item based recommendations
item_item_rmse = accuracy.rmse(item_item_predictions)

print("User-Item based RMSE:", user_item_rmse)
print("Item-Item based RMSE:", item_item_rmse)
