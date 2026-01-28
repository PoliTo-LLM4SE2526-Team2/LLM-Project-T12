# SemEval2026-AER
Course project for "Large Language Models for Software Engineering 25/26" @ PoliTo. Exploring SemEval 2026 Task 12: Abductive Event Reasoning (AER) to investigate real-world causal inference capabilities in LLM.

## Project Structure

```text
AER-Project/
│
├── data/               # Dataset files (SemEval 2026)
├── paper/              # Project report with ACL template
├── src/
│   ├── approaches.py   # Reasoning logic (Baseline, CoT, etc.)
│   ├── dataloader.py   # Data preprocessing and loading
│   ├── evaluator.py    # Evaluate model performance
│   ├── prompts.py      # different type of prompts
│   ├── llm.py          # LLM API Wrapper (DeepSeek)
│   └── retriever.py    # Retrieves and ranks documents
├── requirements.txt    # python dependencies
└── run.py              # Main entry point for experiments
```

## Quick Start
1. Install dependencies
```
> pip install -r requirements.txt
```

