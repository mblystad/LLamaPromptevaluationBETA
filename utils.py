import os
from datetime import datetime

def save_results(prompt: str, results: dict, evaluations: dict, output_dir: str = "."):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"results_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== Prompt ===\n{prompt}\n\n")
        f.write("=== AI Responses by Technique ===\n\n")

        for technique, data in results.items():
            f.write(f"--- {technique.upper()} ---\n")
            for idx, resp in enumerate(data['responses'], 1):
                f.write(f"[Response {idx}]:\n{resp.strip()}\n\n")
            f.write(">>> Analysis:\n")
            for key, val in data['analysis'].items():
                f.write(f"- {key.replace('_', ' ').capitalize()}: {val:.2f}\n")
            f.write(f"- Jaccard Consistency: {data['consistency']:.2f}\n\n")

        f.write("=== QUALITATIVE EVALUATION ===\n\n")
        f.write(evaluations['llm_evaluation'].strip() + "\n")

    print(f"Results saved to {filepath}")
