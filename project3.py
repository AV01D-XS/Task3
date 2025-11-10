from flask import Flask, render_template, request
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

df = pd.read_csv("/home/ai/project_3/travel_destinations_updated.csv")

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

df['description'] = df['description'].apply(clean_text)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df['embeddings'] = df['description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def recommend_places(query, top_n=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = [util.cos_sim(query_emb, emb).item() for emb in df['embeddings']]
    df['score'] = scores
    results = df.sort_values('score', ascending=False).head(top_n)
    return results[['Country', 'City', 'description']]

def safe_recommend(query, top_n=5):
    if not isinstance(query, str) or query.strip() == "":
        return "Please enter a valid query."
    
    if not any(df['Country'].str.contains(query, case=False, na=False)):
        return "Please enter a valid country name."
    
    return recommend_places(query, top_n)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    message = ""
    
    if request.method == "POST":
        query = request.form.get("country", "")
        output = safe_recommend(query, top_n=5)
        
        if isinstance(output, str):  
            message = output
        else:
            results = output.values.tolist()  
    
    return render_template("index.html", results=results, message=message)

if __name__ == "__main__":
    app.run(debug=True)
