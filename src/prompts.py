"""
Centralized prompt storage for all approaches.
Each prompt has a unique name with 'intro', 'system' and 'user' fields.
"""

PROMPTS = {
    "cot": {
        "intro": "The basic zero-shot CoT approach",
        "system_prompt": "You are an expert detective and logic analyst. Your task is Abductive Reasoning: identifying the most plausible cause for an event based on incomplete evidence.",
        "user_prompt": """
        Target Event:
        {event}

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
        
        If there is an option states "None of the others are correct causes." and you have clear evidence that NONE of other options are plausible causes according to what you've retrieved, then choose only this one.

        CRITICAL:: 
        1. There is guaranteed to be AT LEAST one correct answer from the given options, so an empty answer is NOT allowed!
        2. The "Final Answer I Reasoned: ..." line MUST be the very last line of your response. Do NOT write anything after it!
        """
    },
    "optimized": {
        "intro": """
        Optimized approach addressing identified failure patterns:
        1. Under-selection: Structured per-option evaluation
        2. Over-selection: Strict temporal causation check
        3. Cause/Consequence confusion: Explicit directionality check
        4. "None correct" failures: Balanced handling without bias
        5. Duplicate options: Explicit duplicate detection
        """,
        "system_prompt": """
        You are an expert in causal reasoning and abductive inference. Your task is to identify which candidate option(s) are plausible CAUSES of a target event.

        CRITICAL DEFINITIONS:
        - CAUSE: An event/action that happened BEFORE the target event AND directly led to or enabled the target event
        - NOT A CAUSE: Events that happened AFTER, are consequences OF, or are merely correlated with the target event
        - Temporal Rule: A cause must precede its effect in time
        """,
        "user_prompt": """
        TARGET EVENT (the effect we need to explain):
        {event}

        EVIDENCE DOCUMENTS:
        {docs_text}

        CANDIDATE CAUSES:
        {options_text}

        === ANALYSIS INSTRUCTIONS ===

        STEP 1: DUPLICATE CHECK
        First, identify if any options have identical or nearly identical wording. List any duplicates found.

        STEP 2: PER-OPTION CAUSAL ANALYSIS
        For EACH option (A, B, C, D), answer these questions:
        1. TEMPORAL: Did this happen BEFORE the target event? (Yes/No/Unclear)
        2. DOCUMENTED: Is there evidence in the documents supporting this? (Yes/No)
        3. CAUSAL LINK: Does this logically LEAD TO or ENABLE the target event? (Yes/No)
        4. VERDICT: Is this a plausible cause? (CAUSE / NOT_CAUSE / INSUFFICIENT_INFO)

        Format each option analysis as:
        [Option X]: <option text>
        - Temporal: <Yes/No/Unclear>
        - Documented: <Yes/No>
        - Causal Link: <Yes/No>
        - Verdict: <CAUSE/NOT_CAUSE/INSUFFICIENT_INFO>
        - Reasoning: <brief explanation>

        STEP 3: HANDLE "NONE OF THE OTHERS" OPTION
        If one option states "None of the others are correct causes" or similar:
        - This option should be selected ONLY IF all other options received NOT_CAUSE verdict
        - This option should NOT be selected if ANY other option is a valid cause

        STEP 4: FINAL SELECTION RULES
        - Select ALL options with CAUSE verdict
        - If options are duplicates (same/similar text), select ALL duplicate labels
        - If no option qualifies as CAUSE and there's a "None correct" option, select it
        - Never select both regular causes AND "None correct" option

        === OUTPUT FORMAT ===
        After your analysis, state your final answer EXACTLY as:
        "Final Answer I Reasoned: X" (single answer) or "Final Answer I Reasoned: X,Y,Z" (multiple answers, comma-separated, no spaces after commas)

        CRITICAL: 
        1. There is guaranteed to be AT LEAST one correct answer from the given options, so an empty answer is NOT allowed!
        2. The "Final Answer I Reasoned: ..." line MUST be the absolute last line of your response. Do NOT add any text, explanation, or punctuation after it!
        """
    },
    "twopass": {
        "intro": """
        Two-pass reasoning approach:
        Pass 1: Identify all potentially related options (liberal selection)
        Pass 2: Verify causal direction and filter (strict verification)
        """,
        "system_prompt": "You are an expert in abductive reasoning. You will analyze candidate causes for an event using a two-pass verification process.",
        "user_prompt": """
        TARGET EVENT: 
        {event}

        DOCUMENTS:
        {docs_text}

        OPTIONS:
        {options_text}

        === TWO-PASS ANALYSIS ===

        **PASS 1: CANDIDATE IDENTIFICATION (Be Inclusive)**
        For each option, determine if it has ANY connection to the target event based on the documents.
        Mark as CANDIDATE if there's any potential causal relationship. Mark as REJECT only if clearly unrelated.

        Options to consider as candidates: [List A/B/C/D that pass]

        **PASS 2: CAUSAL VERIFICATION (Be Strict)**
        For each CANDIDATE from Pass 1, verify:

        Q1: Does the evidence show this happened BEFORE the target event?
        Q2: Is there a logical mechanism by which this CAUSED or ENABLED the target event?
        Q3: Is this a CAUSE (led to event) or a CONSEQUENCE (resulted from event)?

        Only options answering: Q1=Yes, Q2=Yes, Q3=CAUSE are valid.

        **DUPLICATE HANDLING:**
        If two options have identical text, both labels must be included in the answer.

        **"NONE CORRECT" HANDLING:**
        If an option states "None of the others are correct":
        - Select it ONLY if Pass 2 produces zero valid causes
        - Do NOT select it alongside other causes

        **FINAL ANSWER:**
        List all options that passed both passes.

        State exactly: "Final Answer I Reasoned: X" or "Final Answer I Reasoned: X,Y,Z"

        CRITICAL: 
        1. There is guaranteed to be AT LEAST one correct answer from the given options, so an empty answer is NOT allowed!
        2. This final answer MUST be the absolute last line of your response. Do NOT add any text after it!
        """
    },
    "structured": {
        "intro": """
        Structured Chain-of-Thought with explicit scoring rubric.
        Uses numerical scoring to reduce ambiguity in multi-answer selection.
        """,
        "system_prompt": "You are an expert causal analyst. Score each candidate cause using a structured rubric, then select all options meeting the threshold.",
        "user_prompt": """
        TARGET EVENT: 
        {event}

        EVIDENCE:
        {docs_text}

        CANDIDATE CAUSES:
        {options_text}

        === SCORING RUBRIC ===

        Score each option from 0-3 on each criterion:

        **TEMPORAL (0-3)**
        - 0: Clearly happened AFTER the event (consequence, not cause)
        - 1: Timing unclear or simultaneous
        - 2: Likely before the event
        - 3: Definitely before the event (documented)

        **EVIDENCE (0-3)**
        - 0: No document mentions this
        - 1: Vaguely related to documents
        - 2: Partially supported by documents
        - 3: Directly stated or strongly implied in documents

        **CAUSATION (0-3)**
        - 0: No causal connection to the event
        - 1: Correlated but not causal
        - 2: Contributing factor
        - 3: Direct cause or necessary precondition

        === SCORING TABLE ===

        | Option | Temporal | Evidence | Causation | Total | Select? |
        |--------|----------|----------|-----------|-------|---------|
        | A      |          |          |           |       |         |
        | B      |          |          |           |       |         |
        | C      |          |          |           |       |         |
        | D      |          |          |           |       |         |

        SELECTION RULES:
        - Total >= 7: Strong candidate (SELECT)
        - Total 5-6: Moderate candidate (SELECT if no better options or ties for meaning with a selected option)
        - Total < 5: Weak candidate (DO NOT SELECT)
        - If an option says "None correct" and all others score < 5, select only that option
        - If options have identical text, they get identical scores - select ALL matching labels

        === OUTPUT ===

        1. Fill the scoring table with reasoning
        2. Apply selection rules
        3. State: "Final Answer I Reasoned: X" or "Final Answer I Reasoned: X,Y,Z"

        CRITICAL: 
        1. There is guaranteed to be AT LEAST one correct answer from the given options, so an empty answer is NOT allowed!
        2. The "Final Answer I Reasoned: ..." line MUST be the absolute last line of your response. Do NOT add any text after it!
        """
    }
}