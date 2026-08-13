import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "UrbanBot is a Smart City Analytics Platform.",
    "UrbanBot provides traffic congestion prediction.",
    "UrbanBot provides accident detection.",
    "UrbanBot provides streetlight detection.",
    "UrbanBot provides air quality prediction.",
    "UrbanBot provides crowd density analysis.",
    "UrbanBot provides citizen complaint analysis.",
    "UrbanBot provides smart alerts.",
    "UrbanBot uses machine learning and deep learning models for smart city analytics.",
]

vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(documents).toarray().astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def retrieve_context(question, top_k=3):
    query_vector = vectorizer.transform([question]).toarray().astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []

    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])

    return results


if __name__ == "__main__":

    question = input("Ask UrbanBot: ")

    results = retrieve_context(question)

    print("\nRelevant UrbanBot Information:\n")

    for result in results:
        print("-", result)
