import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import dask.dataframe as dd
from flask_caching import Cache
from clean import dish_recommender  

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Load data lazily using Dask (updated with dtype specification)
ddf = dd.concat([
    dd.read_csv(f"cleaned_file-{i}.csv", dtype={'index': 'float64'}).set_index('index') 
    for i in range(1, 4)
])

# Updated route to /recommend
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()

        if 'ingredients' in data:
            # Ingredient-based recommendation (using your dish_recommender function)
            user_ingredients = data['ingredients']
            if not isinstance(user_ingredients, str):
                return jsonify({'error': 'Invalid ingredients format'}), 400

            recommendations = dish_recommender(user_ingredients)  
            if isinstance(recommendations, str):
                return recommendations   # If it's already a JSON string, return as is
            else:
                return jsonify(recommendations)

        else:
            return jsonify({'error': 'Invalid request data'}), 400

    except KeyError as e:
        app.logger.error(f"Missing key in request data: {e}")
        return jsonify({'error': 'Missing required fields'}), 400
    except pd.errors.EmptyDataError as e:
        app.logger.error(f"No data found: {e}")
        return jsonify({'error': 'No matching recipes found'}), 404
    except Exception as e:
        app.logger.error(f"Error recommending dishes/recipes: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0") 
