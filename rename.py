from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os

def load_data():
    with open("data/human_samples.txt", "r", encoding="utf-8") as f:
        human_texts = [line.strip() for line in f if line.strip()]
    with open("data/ai_samples.txt", "r", encoding="utf-8") as f:
        ai_texts = [line.strip() for line in f if line.strip()]
    return human_texts, ai_texts

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    human_texts, ai_texts = load_data()
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    print("Encoding texts...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)

    print("Training model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/detector.pkl")
    print("Model saved to model/detector.pkl")

if __name__ == "__main__":
    main()
