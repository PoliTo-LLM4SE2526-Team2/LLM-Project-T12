from abc import ABC, abstractmethod
from src.llm import BaseLLM
from src.retriever import DocumentRetriever
from src.dataloader import AERItem
from collections import Counter
import re
from src.prompts import PROMPTS


class BaseApproach(ABC):
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def solve(self, item: AERItem, prompt_name: str) -> str:
        pass


class BaselineApproach(BaseApproach):
    """
    The basic zero-shot CoT approach.
    """

    def solve(self, item: AERItem, prompt_name: str = "cot") -> str:
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents)
            if self.retriever
            else item.documents
        )

        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )

        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.generate(messages)
        return response



class SelfConsistencyRefinementApproach(BaseApproach):
    """
    Combines Self-Consistency (multiple sampling + voting) with Self-Refinement (critique and improve).
    
    Process:
    1. Self-Consistency: Generate multiple reasoning paths with higher temperature
    2. Vote on the most common answer
    3. Self-Refinement: Critique the voted answer and refine it
    """
    
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 5
        self.temperature = 0.7
    
    def _parse_answer_from_response(self, response: str) -> set:
        """Extract answer options from LLM response."""
        if not response:
            return set()
        
        # Try to find "Final Answer I Reasoned: ..." pattern
        pattern = r"Final Answer I Reasoned:\s*([A-D,\s]+)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            answer_str = match.group(1).strip()
            answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
            return {a for a in answers if a in ["A", "B", "C", "D"]}
        
        # Fallback: find any A-D letters
        pattern2 = r"\b([A-D])\b"
        matches = re.findall(pattern2, response[-200:])
        if matches:
            return {m.upper() for m in matches if m.upper() in ["A", "B", "C", "D"]}
        
        return set()
    
    def _get_prompt(self, item: AERItem, prompt_name: str) -> tuple:
        """Get the system and user prompts."""
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )

        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )
        
        return system_prompt, user_prompt, docs_text, options_text, item.event
    
    def solve(self, item: AERItem, prompt_name: str = "cot") -> str:
        """
        Main solving method combining Self-Consistency and Self-Refinement.
        """
        system_prompt, user_prompt, docs_text, options_text, event = self._get_prompt(item, prompt_name)
        
        # ============ STAGE 1: Self-Consistency ============
        print(f"\n[Self-Consistency] Generating {self.num_samples} reasoning paths...")
        
        all_responses = []
        all_answers = []
        
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=self.temperature)
            all_responses.append(response)
            
            # Parse answer
            answer = self._parse_answer_from_response(response)
            all_answers.append(frozenset(answer))  # Use frozenset for hashable voting
            
            print(f"  Sample {i+1}: {sorted(list(answer)) if answer else 'No answer'}")
        
        # Vote for the most common answer
        if not all_answers or all(len(a) == 0 for a in all_answers):
            # If no valid answers, return first response
            return all_responses[0] if all_responses else ""
        
        answer_counts = Counter(all_answers)
        most_common_answer, vote_count = answer_counts.most_common(1)[0]
        most_common_answer = set(most_common_answer)
        
        print(f"\n[Voting Result] Most common answer: {sorted(list(most_common_answer))} (votes: {vote_count}/{self.num_samples})")
        
        # Find the response that matches the most common answer
        best_response = None
        for response in all_responses:
            if self._parse_answer_from_response(response) == most_common_answer:
                best_response = response
                break
        
        if not best_response:
            best_response = all_responses[0]
        
        # ============ STAGE 2: Self-Refinement ============
        print("\n[Self-Refinement] Critiquing and refining the answer...")
        
        critique_prompt = f"""
        You previously analyzed this abductive reasoning problem and concluded:
        
        {best_response}
        
        Now, critically review your reasoning:
        1. Are there any logical flaws or inconsistencies in your analysis?
        2. Did you miss any important evidence from the documents?
        3. Did you correctly apply the rule about selecting ALL plausible causes?
        4. For the "None of the others" option, did you verify that truly NONE of the other options are plausible?
        5. Are there any options you selected without sufficient evidence, or rejected despite having support?
        
        If you find issues, provide a critique. If your reasoning is sound, confirm it.
        """
        
        critique_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt}
        ]
        
        critique_response = self.llm.generate(critique_messages, temperature=0.3)
        
        # Generate refined answer
        refinement_prompt = f"""
        Target Event:
        {event}

        Retrieved Evidence:
        {docs_text}

        Candidate Causes:
        {options_text}

        Your initial reasoning:
        {best_response}
        
        Your self-critique:
        {critique_response}
        
        Based on your critique, provide your FINAL refined answer. If your initial reasoning was correct, you can confirm it. If there were issues, correct them.
        
        Output format:
        First, briefly explain any changes you're making (or confirm your original answer).
        Then, provide your detailed final reasoning.
        Finally, state the answer strictly in this format: "Final Answer I Reasoned: [Option Label]".
        
        Remember:
        - Select ALL options that are plausible causes supported by evidence
        - Only choose "None of the others" if you have clear evidence that ALL other options are incorrect
        """
        
        refinement_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refinement_prompt}
        ]
        
        final_response = self.llm.generate(refinement_messages, temperature=0.1)
        
        print("[Self-Refinement] Complete!")
        
        # Return the full refinement process for transparency
        full_output = f"""
        ========== SELF-CONSISTENCY STAGE ==========
        Generated {self.num_samples} samples, voted answer: {sorted(list(most_common_answer))} ({vote_count}/{self.num_samples} votes)

        Best reasoning from consistency stage:
        {best_response}

        ========== SELF-REFINEMENT STAGE ==========
        Self-Critique:
        {critique_response}

        ========== FINAL REFINED ANSWER ==========
        {final_response}
        """
        
        return full_output
