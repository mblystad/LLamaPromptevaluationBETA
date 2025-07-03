
from langchain_ollama import OllamaLLM

# Load local Ollama model (GPU enabled if supported)
llm = OllamaLLM(model="llama3")

def basic_prompt_chain(prompt: str) -> str:
    return llm.invoke(prompt)

def few_shot_chain(prompt: str, examples: list[str]) -> str:
    formatted = "\n".join(f"Input: {ex}\nOutput: ..." for ex in examples)
    full_prompt = f"{formatted}\nInput: {prompt}\nOutput:"
    return llm.invoke(full_prompt)

def chain_of_thought_chain(prompt: str) -> str:
    cot_prompt = f"{prompt}\nLet's think step by step:"
    return llm.invoke(cot_prompt)

def tree_of_thought_chain(prompt: str) -> str:
    branches = [
        f"{prompt}\nOption 1: Let's consider a logical approach...",
        f"{prompt}\nOption 2: What if we take a creative perspective...",
        f"{prompt}\nOption 3: Suppose we reverse the assumptions..."
    ]
    outputs = [llm.invoke(b) for b in branches]
    return "\n---\n".join(outputs)

def self_consistency_chain(prompt: str, runs: int = 1) -> list[str]:
    return [llm.invoke(prompt) for _ in range(runs)]
