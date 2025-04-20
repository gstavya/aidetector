from sentence_transformers import SentenceTransformer
import joblib
import sys

def predict(text, show_chunks=False):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    clf = joblib.load("model/detector.pkl")

    chunks = text.split("\n\n")
    chunk_embeddings = model.encode(chunks)

    preds = clf.predict_proba(chunk_embeddings)
    ai_probs = [p[1] for p in preds]

    doc_level_score = sum(ai_probs) / len(ai_probs)
    print(f"\nDocument AI-likeness Score: {doc_level_score:.2f}")

    if show_chunks:
        for i, prob in enumerate(ai_probs):
            print(f"\n--- Chunk {i+1} ---")
            print(chunks[i][:300], "..." if len(chunks[i]) > 300 else "")
            print(f"AI-likeness: {prob:.2f}")

if __name__ == "__main__":
    text_input = input()
    predict(text_input, show_chunks=True)
