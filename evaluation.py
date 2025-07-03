import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
from langchain_community.llms import Ollama

llm_eval = Ollama(model="llama3")

def analyze_response(text):
    words = text.split()
    unique_words = set(words)
    return {
        "length": len(text),
        "word_count": len(words),
        "lexical_richness": len(unique_words) / len(words) if words else 0,
        "flesch_reading_ease": textstat.flesch_reading_ease(text)
    }

def jaccard_consistency(responses):
    vectorizer = CountVectorizer(binary=True)
    try:
        X = vectorizer.fit_transform(responses).toarray()
    except ValueError:
        return 0.0
    if len(X) < 2:
        return 1.0
    scores = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            score = jaccard_score(X[i], X[j])
            scores.append(score)
    return np.mean(scores) if scores else 0.0

def evaluate_best_response(prompt, results_dict):
    system_prompt = (
        "You are an expert writing coach evaluating AI responses to a prompt. "
        "Rank the different answers in terms of helpfulness, clarity, depth, and completeness. "
        "Be critical, comparative, and provide a reasoned explanation of which is best."
    )

    combined_input = f"Prompt: {prompt}\n"
    for technique, data in results_dict.items():
        combined_input += f"\nTechnique: {technique}\n"
        for idx, resp in enumerate(data["responses"], 1):
            combined_input += f"[{idx}] {resp}\n"

    return llm_eval.invoke(system_prompt + "\n" + combined_input)
