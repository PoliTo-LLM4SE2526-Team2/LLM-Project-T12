from abc import ABC, abstractmethod
from src.llm import BaseLLM
from src.dataloader import AERItem

class BaseSolver(ABC):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def solve(self, item: AERItem) -> str:
        pass

class BaselineSolver(BaseSolver):
    """
    The basic zero-shot CoT solver.
    """
    def solve(self, item: AERItem) -> str:
        docs_text = "\n".join(f"Document{i+1}: {doc}" for i, doc in enumerate(item.documents))
        options_text = "\n".join(f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options))

        system_prompt = "You are an expert detective and logic analyst. Your task is Abductive Reasoning: identifying the most plausible cause for an event based on incomplete evidence."

        user_prompt = f"""
        Target Event:
        {item.event}

        Retrieved Evidence:
        {docs_text}

        Candidate Causes:
        {options_text}

        Instruction:
        1. Analyze the relationship between the event and the documents.
        2. Evaluate each candidate cause.
        3. Select the most plausible cause.

        Output format:
        First, provide a brief reasoning chain point by point.
        Finally, output the answer strictly in this format: "Final Answer I Reasoned: [Option Label]".
        Note that you have to output all satisfied labels, for example the final answer you reasoned is A:
        1. if you find B and C have the same content with A, then you have to output A,B,C.
        2. if you find B express the same meaning with A but just with a different way of saying it, then you have to output A,B.
        3. if you find C encompassed by A, then you have to output A,C.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.generate(messages)
        return response
