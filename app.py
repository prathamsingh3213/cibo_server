from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
# tfidf_vectorizer = load('tfidf_vectorizer.joblib')
# tfidf_matrix = load('tfidf_matrix.joblib')

@app.route('/recommend', methods=['POST'])
def recommend_dish():
    try:
        def filter_ingredients(df, filter_list):
            filtered_df = df[df['ingredients'].apply(lambda x: all(ingredient in x for ingredient in filter_list))]
            return filtered_df
        print("Inside the function")
        data = request.get_json()
        ingredients = data['ingredients']
        essentials=data['essentials']
        df1=pd.read_csv("cleaned_file-1.csv.gz",index_col="index")
        df2=pd.read_csv("cleaned_file-2.csv.gz",index_col="index")
        df3=pd.read_csv("cleaned_file-3.csv.gz",index_col="index")
        df=pd.concat([df1,df2,df3])

        result=filter_ingredients(df.copy(),essentials)
        # result['ingredients']


        tfidf_vectorizer2=TfidfVectorizer(stop_words='english')
        tfidf_matrix2=tfidf_vectorizer2.fit_transform(result['ingredients'])


        # Use the model for recommendation
        user_idf = tfidf_vectorizer2.transform([ingredients])
        sim_ing = cosine_similarity(user_idf, tfidf_matrix2)
        li=sorted(list(enumerate(sim_ing[0])),reverse=True,key=lambda x:x[1])[0:5]
        li
        indices = [index for index, _ in li]
        newdf=result.iloc[indices]
        json=newdf.to_json(orient='records')

        # Use the model for recommendation
        user_idf = tfidf_vectorizer2.transform([ingredients])
        sim_ing = cosine_similarity(user_idf, tfidf_matrix2)
        li=sorted(list(enumerate(sim_ing[0])),reverse=True,key=lambda x:x[1])[0:5]
        li
        indices = [index for index, _ in li]
        newdf=df.loc[indices]
        json=newdf.to_json(orient='records')
        return json

    except Exception as e:
        # Log the exception details
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=False)
