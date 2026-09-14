from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)
def clean_text(t):
    if not isinstance(t, str): return ""
    t = re.sub(r"http\S+|[^a-z\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()
def load_and_train():
    t_df = pd.read_csv('True.csv')
    f_df = pd.read_csv('Fake.csv')
    t_df["label"] = 1
    f_df["label"] = 0
    df = pd.concat([t_df, f_df], ignore_index=True).dropna(subset=["title", "text"])
    df["combined"] = df["title"].apply(clean_text) + " " + df["text"].apply(clean_text)
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(df["combined"])
    y = df["label"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return vec, clf
vectorizer, model = load_and_train()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_text = request.form.get('news_text')
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        confidence = round(max(prob) * 100, 2)
        result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
        return render_template('index.html', prediction_text=f'{result} ({confidence}%)')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")
if __name__ == '__main__':
    app.run(debug=True)