from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ✅ Corrected dataset path
df = pd.read_csv("/home/ai/Project-3A/travel_destinations_updated.csv")

# 🧹 Clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

df['description'] = df['description'].apply(clean_text)

# 🧠 Load sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 💾 Generate embeddings (NumPy instead of Torch)
df['embeddings'] = df['description'].apply(lambda x: model.encode(x, convert_to_numpy=True))

# 📏 Compute cosine similarity manually
def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b)

# 🧭 Recommend places
def recommend_places(query, top_n=5):
    query_emb = model.encode(query, convert_to_numpy=True)
    scores = [cosine_similarity(query_emb, emb) for emb in df['embeddings']]
    df['score'] = scores
    results = df.sort_values('score', ascending=False).head(top_n)
    return results[['Country', 'City', 'description']]

# 🔍 Validate and handle bad input
def safe_recommend(query, top_n=5):
    if not isinstance(query, str) or query.strip() == "":
        return "Please enter a valid query."
    
    if not any(df['Country'].str.contains(query, case=False, na=False)):
        return "Please enter a valid country name."
    
    return recommend_places(query, top_n)

# 🌍 Flask routes
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
