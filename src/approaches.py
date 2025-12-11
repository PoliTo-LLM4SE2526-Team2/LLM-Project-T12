from abc import ABC, abstractmethod
from src.llm import BaseLLM
from src.retriever import DocumentRetriever
from src.dataloader import AERItem

class BaseApproach(ABC):
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def solve(self, item: AERItem) -> str:
        pass

class BaselineApproach(BaseApproach):
    """
    The basic zero-shot CoT approach.
    """
    def solve(self, item: AERItem) -> str:
        documents = self.retriever.retrieve(item.event, item.documents) if self.retriever else item.documents

        docs_text = "\n".join(f"Document{i+1}: {doc}" for i, doc in enumerate(documents))
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
        3. Select the plausible cause(s).

        Output format:
        First, provide a detailed reasoning chain explaining:
        1. What information is found in the documents related to the event.
        2. How each candidate cause relates to the event.
        3. Why certain causes are more plausible than others.
        4. Which documents support or contradict each option.
        5. Your final conclusion with clear justification.
        Finally, state the answer strictly in this format: "Final Answer I Reasoned: [Option Label]".
        Your output must strictly adhere to the format and order specified above!!!

        Note that there may be one or multiple correct option(s), you have to select ALL options that are directly supported or strongly implied by the documents as plausible causes of the event, for example the final answer you reasoned is A:
        1. if you find B and C have the same content with A, then you have to output A,B,C.
        2. if you find B express the same meaning with A but just with a different way of saying it, then you have to output A,B.
        3. if you find C encompassed by A, then you have to output A,C.

        If there is an option states "None of the others are correct causes." and you have clear evidence that NONE of other options are plausible causes according to what you've retrieved, then choose only this one. Otherwise, never choose this option.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.generate(messages)
        return response
