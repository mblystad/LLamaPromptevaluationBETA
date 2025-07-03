from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def load_llm(model_name="tiiuae/Falcon3-3B-Instruct", max_tokens=500, use_gpu=False):
    """
    Loads a HuggingFace model as a LangChain-compatible LLM.

    Args:
        model_name (str): Name of the model on Hugging Face Hub.
        max_tokens (int): Maximum number of tokens to generate.
        use_gpu (bool): If True, use GPU (device=0). If False, use CPU (device=-1).

    Returns:
        LangChain-compatible LLM object.
    """
    print(f"[INFO] Loading model: {model_name}")

    device = 0 if use_gpu else -1  # -1 means CPU

    # Load the transformers pipeline for text-generation
    hf_pipeline = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        max_new_tokens=max_tokens,
        return_full_text=False  # We only want generated output, not the full input+output
    )

    # Wrap the pipeline in LangChain interface
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print("[INFO] Model loaded and wrapped in LangChain pipeline.")
    return llm
