# LLamaPromptevaluationBETA
A beta version of an LLM based prompt-technique evaluator:
Python CLI tool for exploring and evaluating prompt engineering techniques using a local LLM via LangChain and Ollama. Designed for students, researchers, and hobbyists who want to **experiment with structured prompting strategies** and see how different chains affect the quality of LLM output.

---

## 🔍 Features

- Compare outputs from multiple prompting techniques:
  - `basic`
  - `few-shot`
  - `chain-of-thought`
  - `tree-of-thought`
  - `self-consistency`
  - and more...
- Automatically evaluate responses using a local LLM.
- Output includes:
  - Full response set by technique
  - LLM-based qualitative ranking
  - A declared "🏆 Best Technique"
  - Readability and consistency metrics
- Modular code structure for easy extension.

---

## ⚙️ Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A supported local model like `llama3` pulled via Ollama:
```

ollama pull llama3

````

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/prompt-engineering-cli.git
cd prompt-engineering-cli
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
````

---

## 🚀 Usage

Run the main CLI script:

```bash
python main.py
```

You'll be prompted to enter:

1. The base prompt (e.g., "Give a summary of Crime and Punishment")
2. A role/persona (e.g., "Be a Russian literary critic")
3. A few-shot hint string (e.g., "symbolism: morality: psychology")

All chains will generate responses and write the outputs + evaluation to:

* `evaluation_summary.txt`

---

## 📁 Project Structure

```
.
├── chains.py          # All LangChain prompt chains
├── evaluation.py      # Qualitative + consistency evaluation logic
├── utils.py           # Scoring and formatting utilities
├── main.py            # CLI entry point
├── evaluation_summary.txt  # Output file
└── README.md
```

---

## ✏️ Customization

* 🔄 **To add your own technique**, edit `chains.py` and register it in `main.py`.
* 💬 **Change the evaluation logic** in `evaluation.py` to use your own rubric.
* 🧪 **Swap the model** via the `Ollama(model="...")` line in `chains.py`.

---

## 🤝 Credits

Created by [Magnus Helgheim Blystad](https://github.com/mblystad)


---

## 🧠 Philosophy

> Prompting is not a trick. It’s a method of thinking.
> This tool helps you **see prompting as reasoning** – with outputs you can measure, test, and refine.

---

## 📜 License

MIT License

