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
        user_ingredients = data['ingredients'].lower().split(',')
        essentials = [ess.lower() for ess in data.get("essentials", [])]

        # Filter by essentials 
        filtered_df = df[df['ingredients'].apply(lambda x: all(ess in x for ess in essentials))]

        # If no essentials, use the pre-trained model
        if not essentials:
            tfidf_vectorizer_to_use = tfidf_vectorizer
            tfidf_matrix_to_use = tfidf_matrix
        else:
            # Re-train vectorizer only on filtered data if essentials provided
            tfidf_vectorizer_to_use = TfidfVectorizer(stop_words='english') 
            tfidf_matrix_to_use = tfidf_vectorizer_to_use.fit_transform(filtered_df['ingredients'])

        user_tfidf = tfidf_vectorizer_to_use.transform([','.join(user_ingredients)])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix_to_use)
        top_indices = similarities[0].argsort()[-8:][::-1] 

        recommendations = filtered_df.iloc[top_indices].to_json(orient='records')
        return recommendations
    
    except Exception as e:
        app.logger.error(f"Error recommending dishes: {e}") 
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=False)
