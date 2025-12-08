# SemEval2026-AER
Course project for "Large Language Models for Software Engineering 25/26" @ PoliTo. Exploring SemEval 2026 Task 12: Abductive Event Reasoning (AER) to investigate real-world causal inference capabilities in LLM.

## Project Structure

```text
AER-Project/
│
├── data/               # Dataset files (SemEval 2026)
├── paper/              # Project report with ACL template
├── results/            # Model training results
├── src/
│   ├── approaches.py   # Reasoning logic (Baseline, CoT, etc.)
│   ├── dataloader.py   # Data preprocessing and loading
│   ├── evaluator.py    # Evaluate model performance
│   └── llm.py          # LLM API Wrapper (DeepSeek)
└── run.py              # Main entry point for experiments
