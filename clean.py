from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import dask.dataframe as dd  

# Load data lazily using Dask (updated)
ddf = dd.concat([
    dd.read_csv(f"cleaned_file-{i}.csv", dtype={'index': 'float64'}).set_index('index') 
    for i in range(1, 4)
])

# Compute and load the entire DataFrame into memory
df = ddf.compute()

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Calculate the TF-IDF matrix (do this only once)
tfidf_matrix = tfidf_vectorizer.transform(df['ingredients']) 

def dish_recommender(ingrd_str):
    try:
        # Convert string input to list of ingredients
        ingredients = ingrd_str.strip('[]').replace("'", "").split(',')  

        user_tfidf = tfidf_vectorizer.transform([' '.join(ingredients)])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)

        # Get top 5 similar recipe indices
        top_indices = similarities[0].argsort()[-8:][::-1]  

        # Fetch recommendations from the DataFrame
        recommendations = df.iloc[top_indices].to_json(orient='records')
        return recommendations
        
    except ValueError as e:
        # Handle cases where no similar recipes are found
        return jsonify({"error": "No matching recipes found"}), 404
    except Exception as e:
        # Handle other unexpected errors
        return jsonify({"error": str(e)}), 500
