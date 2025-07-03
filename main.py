from chains import (
    basic_prompt_chain,
    few_shot_chain,
    chain_of_thought_chain,
    tree_of_thought_chain,
    self_consistency_chain
)
from evaluation import analyze_response, jaccard_consistency, evaluate_best_response
from utils import save_results

def get_user_inputs():
    print("Enter the main prompt you'd like to test:")
    prompt = input("> ")

    print("\nEnter role or persona for prompt (used by chain-of-thought):")
    role = input("> ")

    print("\nEnter a few examples (used by few-shot), separated by semicolons:")
    examples_input = input("> ")
    examples = [ex.strip() for ex in examples_input.split(";") if ex.strip()]

    return prompt, role, examples

def main():
    prompt, role, examples = get_user_inputs()

    results = {}

    # === BASIC ===
    basic_response = basic_prompt_chain(prompt)
    results["basic"] = {
        "responses": [basic_response],
        "analysis": analyze_response(basic_response),
        "consistency": 1.0
    }

    # === FEW-SHOT ===
    few_shot_response = few_shot_chain(prompt, examples)
    results["few_shot"] = {
        "responses": [few_shot_response],
        "analysis": analyze_response(few_shot_response),
        "consistency": 1.0
    }

    # === CHAIN OF THOUGHT ===
    cot_prompt = f"You are a {role}. {prompt}"
    cot_response = chain_of_thought_chain(cot_prompt)
    results["chain_of_thought"] = {
        "responses": [cot_response],
        "analysis": analyze_response(cot_response),
        "consistency": 1.0
    }

    # === TREE OF THOUGHT ===
    tree_response = tree_of_thought_chain(prompt)
    results["tree_of_thought"] = {
        "responses": [tree_response],
        "analysis": analyze_response(tree_response),
        "consistency": 1.0  # Not repeated
    }

    # === SELF CONSISTENCY ===
    sc_responses = self_consistency_chain(prompt)
    joined_responses = "\n\n".join(sc_responses)
    avg_analysis = analyze_response(joined_responses)
    consistency_score = jaccard_consistency(sc_responses)
    results["self_consistency"] = {
        "responses": sc_responses,
        "analysis": avg_analysis,
        "consistency": consistency_score
    }

    evaluation = {
        "llm_evaluation": evaluate_best_response(prompt, results)
    }

    save_results(prompt, results, evaluation)

if __name__ == "__main__":
    main()
