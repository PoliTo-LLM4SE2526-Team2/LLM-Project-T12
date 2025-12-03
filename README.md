# SemEval2026-AER
Course project for "Large Language Models for Software Engineering" @ PoliTo. Exploring SemEval 2026 Task 12: Abductive Event Reasoning (AER) to investigate and enhance real-world causal inference capabilities in LLM.

## Project Structure

```text
AER-Project/
├── data/               # Dataset files (SemEval 2026)
├── src/
│   ├── llm.py          # LLM API Wrapper (DeepSeek)
│   ├── dataloader.py   # Data preprocessing and loading
│   └── solvers.py      # Reasoning logic (Baseline, CoT, etc.)
└── run.py              # Main entry point for experiments