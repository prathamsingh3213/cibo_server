from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
# tfidf_vectorizer2 = load("path/to/tfidf_vectorizer.joblib")
# tfidf_matrix2 = load("path/to/tfidf_matrix.joblib")

# Updated filtering function
def filter_ingredients(df, filter_list, essentials=None):
    if essentials:  # If essentials list is provided and not empty
        filtered_df = df[df['ingredients'].apply(lambda x: all(ingredient in x for ingredient in essentials))]
    else:  
        filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['ingredients'].apply(lambda x: all(ingredient in x for ingredient in filter_list))]
    return filtered_df

@app.route('/recommend', methods=['POST'])
def recommend_dish():
    try:
        data = request.get_json()

        # Input validation
        if 'ingredients' not in data:
            return jsonify({'error': 'Missing "ingredients" field'}), 400
        if not isinstance(data['ingredients'], list):
            return jsonify({'error': '"ingredients" must be a list'}), 400

        ingredients = data['ingredients']
        essentials = [essential.lower() for essential in data.get("essentials", [])] 

        # Load and concatenate data (consider moving this outside the function for efficiency)
        df1 = pd.read_csv("cleaned_file-1.csv.gz", index_col="index")
        df2 = pd.read_csv("cleaned_file-2.csv.gz", index_col="index")
        df3 = pd.read_csv("cleaned_file-3.csv.gz", index_col="index")
        df = pd.concat([df1, df2, df3])

        # Filter and recommend
        result = filter_ingredients(df.copy(), ingredients, essentials)  # Updated call
        user_idf = tfidf_vectorizer2.transform([ingredients])
        sim_ing = cosine_similarity(user_idf, tfidf_matrix2)

        li = sorted(list(enumerate(sim_ing[0])), reverse=True, key=lambda x: x[1])[0:5]
        indices = [index for index, _ in li]
        newdf = result.iloc[indices]
        json_data = newdf.to_json(orient='records')

        return json_data
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=False)  
