from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
tfidf_matrix = load('tfidf_matrix.joblib')
df = pd.concat([pd.read_csv(f"cleaned_file-{i}.csv", index_col="index") for i in range(1, 4)])  # Load all DataFrames

@app.route('/recommend', methods=['POST'])
def recommend_dish():
    try:
        data = request.get_json()
        user_ingredients = data['ingredients'].lower().split(',')  # Lowercase & split ingredients
        essentials = [ess.lower() for ess in data.get("essentials", [])]

        # Filter by essentials 
        filtered_df = df[df['ingredients'].apply(lambda x: all(ess in x for ess in essentials))]

        # Re-train vectorizer only on filtered data if essentials provided
        if essentials:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english') 
            tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['ingredients'])

        user_tfidf = tfidf_vectorizer.transform([','.join(user_ingredients)])  # Combine user ingredients
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)
        top_indices = similarities[0].argsort()[-8:][::-1]  # Get top 8 most similar indices

        recommendations = filtered_df.iloc[top_indices].to_json(orient='records')
        return recommendations
    
    except Exception as e:
        app.logger.error(f"Error recommending dishes: {e}")  # Log error for debugging
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
