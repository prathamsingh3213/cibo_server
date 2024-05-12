from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the original DataFrame and models (adjust paths if needed)
df = pd.read_csv("cleaned_file.csv")  
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')

def dish_recommender(ingrd_str):
    try:
        # Convert string input to list of ingredients
        ingredients = ingrd_str.strip('[]').replace("'", "").split(',')  
        
        user_tfidf = tfidf_vectorizer.transform([' '.join(ingredients)])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)

        # Get top 5 similar recipe indices
        top_indices = similarities[0].argsort()[-5:][::-1]  
        
        # Fetch recommendations from the DataFrame
        recommendations = df.iloc[top_indices].to_json(orient='records')
        return recommendations
        
    except ValueError as e:
        # Handle cases where no similar recipes are found
        return jsonify({"error": "No matching recipes found"}), 404
    except Exception as e:
        # Handle other unexpected errors
        return jsonify({"error": str(e)}), 500

# Uncomment this if you're running clean.py separately to test:
# if __name__ == "__main__":
#    ingredients_input = input("Enter ingredients (comma-separated): ")
#    dish_recommender(ingredients_input)
