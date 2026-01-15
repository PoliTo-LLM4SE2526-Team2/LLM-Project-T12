"""
Approaches for SemEval 2026 Task 12: Abductive Event Reasoning

Optimized for evaluation metric:
- 1.0 points: Perfect match
- 0.5 points: Partial match (subset, no wrong)
- 0.0 points: Any wrong selection

Key insight: Conservative strategy is optimal.
"""

from abc import ABC, abstractmethod
from itertools import count

from torch import threshold
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
    
    def _parse_answer_from_response(self, response: str) -> set:
        """Extract answer options from LLM response."""
        if not response:
            return set()
        
        # Try to find "Final Answer I Reasoned: ..." pattern
        pattern = r"Final Answer I Reasoned:\s*([A-D](?:\s*,\s*[A-D])*)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            answer_str = match.group(1).strip()
            answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
            return {a for a in answers if a in ["A", "B", "C", "D"]}
        
        # Fallback: find any A-D letters in last 200 chars
        pattern2 = r"\b([A-D])\b"
        matches = re.findall(pattern2, response[-200:])
        if matches:
            return {m.upper() for m in matches if m.upper() in ["A", "B", "C", "D"]}
        
        return set()


# ============================================================
# 后处理工具函数
# ============================================================

def detect_duplicate_options(options: list) -> list:
    """
    检测重复或几乎相同的选项
    
    Returns:
        list of tuples: [(idx1, idx2, "identical"), ...]
    """
    labels = ["A", "B", "C", "D"]
    duplicates = []
    
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            # 标准化比较：去除首尾空格，转小写
            opt_i = options[i].strip().lower()
            opt_j = options[j].strip().lower()
            
            if opt_i == opt_j:
                duplicates.append((labels[i], labels[j], "identical"))
            # 可以添加更多相似度检测逻辑
    
    return duplicates


def find_none_correct_option(options: list) -> str:
    """
    找到 "None of the others are correct" 类型的选项
    
    Returns:
        选项标签 (A/B/C/D) 或 None
    """
    labels = ["A", "B", "C", "D"]
    none_keywords = ["none of the others", "none of the above", "none are correct"]
    
    for i, opt in enumerate(options):
        opt_lower = opt.lower()
        if any(keyword in opt_lower for keyword in none_keywords):
            return labels[i]
    
    return None


def post_process_answers(answers: set, options: list) -> set:
    """
    后处理答案，强制执行逻辑规则
    
    Rules:
    1. 重复选项必须同时选中或同时不选
    2. "None correct" 不能与其他选项同时选中
    3. 答案不能为空
    """
    if not answers:
        return answers
    
    processed = answers.copy()
    
    # Rule 1: 处理重复选项
    duplicates = detect_duplicate_options(options)
    for label1, label2, dup_type in duplicates:
        # 如果选了其中一个，必须两个都选
        if label1 in processed or label2 in processed:
            processed.add(label1)
            processed.add(label2)
    
    # Rule 2: 处理 "None correct" 互斥性
    none_label = find_none_correct_option(options)
    if none_label and none_label in processed:
        # 如果选了 "None correct"，检查是否还选了其他选项
        other_answers = processed - {none_label}
        if other_answers:
            # 矛盾！根据保守策略，移除 "None correct"（保留实质性答案）
            processed.discard(none_label)
    
    return processed


# ============================================================
# 保守策略 Approach（推荐）
# ============================================================

class ConservativeApproach(BaseApproach):
    """
    Conservative approach optimized for partial matching metric.
    
    Key principle: Better to miss correct answers than select wrong ones.
    - Wrong selection = 0 points
    - Partial correct = 0.5 points
    """
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # 检索相关文档
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        
        # 获取 prompt
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
        
        # 生成回答
        response = self.llm.generate(messages)
        
        # 解析答案
        raw_answers = self._parse_answer_from_response(response)
        
        # 后处理
        processed_answers = post_process_answers(raw_answers, item.options)
        
        # 如果后处理改变了答案，追加说明
        if processed_answers != raw_answers:
            response += f"\n\n[Post-processing applied: {raw_answers} -> {processed_answers}]"
        
        return response


# ============================================================
# 轻量级 Consistency Approach
# ============================================================

class LightweightConsistencyApproach(BaseApproach):
    """
    Lightweight Self-Consistency with option-level voting.
    
    Only 3 samples (not 5+), no refinement step.
    Uses option-level voting instead of answer-set voting.
    """
    
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 3
        self.temperature = 0.5
        self.vote_threshold = 2  # 至少 2/3 才选
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # 检索相关文档
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
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
        
        # ============ 多次采样 ============
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []
        
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=self.temperature)
            all_responses.append(response)
            
            # 解析答案并投票
            answers = self._parse_answer_from_response(response)
            for opt in answers:
                option_votes[opt] += 1
        
        # ============ 选项级投票 ============
        # 只选择超过阈值的选项
        voted_answers = {opt for opt, count in option_votes.items() 
                        if count >= self.vote_threshold}
        
        # 如果没有选项超过阈值，选择得票最高的
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {opt for opt, count in option_votes.items() 
                               if count == max_votes}
        
        # ============ 后处理 ============
        final_answers = post_process_answers(voted_answers, item.options)
        
        # 构建输出
        vote_summary = ", ".join(f"{opt}:{count}" for opt, count in sorted(option_votes.items()))
        
        output = f"""
========== LIGHTWEIGHT CONSISTENCY ==========
Samples: {self.num_samples}, Threshold: {self.vote_threshold}
Vote counts: {vote_summary}
Voted answers: {sorted(final_answers)}

========== BEST RESPONSE ==========
{all_responses[0] if all_responses else "No response"}

Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
"""
        return output


# ============================================================
# Two-Pass Approach（真正的两次调用）
# ============================================================

class TwoPassApproach(BaseApproach):
    """
    True two-pass approach with separate API calls.
    
    Pass 1: Liberal candidate selection (high recall)
    Pass 2: Strict causal verification (high precision)
    """
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # 检索相关文档
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        
        # ============ PASS 1: 宽松筛选 ============
        pass1_system = "You are an expert in causal reasoning. Your task is to identify ALL potentially relevant options."
        pass1_user = f"""
TARGET EVENT: {item.event}

DOCUMENTS:
{docs_text}

OPTIONS:
{options_text}

TASK: For each option, determine if it has ANY potential connection to the target event.
Be INCLUSIVE at this stage - mark as CANDIDATE if there's any possible relationship.

For each option, answer:
- Option A: CANDIDATE or REJECT? (one word)
- Option B: CANDIDATE or REJECT? (one word)
- Option C: CANDIDATE or REJECT? (one word)
- Option D: CANDIDATE or REJECT? (one word)

Then list all CANDIDATE options.
"""
        
        pass1_response = self.llm.generate([
            {"role": "system", "content": pass1_system},
            {"role": "user", "content": pass1_user}
        ], temperature=0.3)
        
        # 解析 Pass 1 候选
        candidates = set()
        for label in ["A", "B", "C", "D"]:
            # 检查该选项是否被标记为 CANDIDATE
            if re.search(rf"Option {label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE):
                candidates.add(label)
            elif re.search(rf"{label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE):
                candidates.add(label)
        
        # 如果没有找到明确的候选，尝试其他解析方式
        if not candidates:
            # 查找 "candidates: A, B" 类似的模式
            match = re.search(r"candidates?[:\s]*([A-D](?:\s*,\s*[A-D])*)", pass1_response, re.IGNORECASE)
            if match:
                candidates = {c.strip().upper() for c in match.group(1).split(",") if c.strip().upper() in ["A", "B", "C", "D"]}
        
        # 如果还是没有，默认所有选项都是候选
        if not candidates:
            candidates = {"A", "B", "C", "D"}
        
        # ============ PASS 2: 严格验证 ============
        candidates_text = ", ".join(sorted(candidates))
        pass2_system = """You are an expert in causal reasoning. Your task is to verify which candidates are TRUE CAUSES.

CRITICAL SCORING RULE:
- Selecting ANY wrong option = 0 points
- Missing some correct options = 0.5 points
- Be CONSERVATIVE: Only select options you are CERTAIN about."""
        
        pass2_user = f"""
TARGET EVENT: {item.event}

DOCUMENTS:
{docs_text}

CANDIDATE OPTIONS (from Pass 1): {candidates_text}

For each candidate, verify:
1. TEMPORAL: Does evidence show this happened BEFORE the target event? (YES/NO)
2. CAUSAL: Is there a clear mechanism by which this CAUSED the event? (YES/NO)
3. EVIDENCE: Is there direct documentary support? (YES/NO)

Only select options with ALL THREE = YES.

Remember: Wrong selection = 0 points. Be conservative!

Final Answer I Reasoned: [Only verified options]
"""
        
        pass2_response = self.llm.generate([
            {"role": "system", "content": pass2_system},
            {"role": "user", "content": pass2_user}
        ], temperature=0.1)
        
        # 解析最终答案
        raw_answers = self._parse_answer_from_response(pass2_response)
        
        # 后处理
        final_answers = post_process_answers(raw_answers, item.options)
        
        # 构建输出
        output = f"""
========== TWO-PASS APPROACH ==========

----- PASS 1: Candidate Selection -----
Candidates identified: {sorted(candidates)}

{pass1_response}

----- PASS 2: Strict Verification -----
{pass2_response}

----- POST-PROCESSING -----
Raw answers: {sorted(raw_answers)}
Final answers: {sorted(final_answers)}

Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
"""
        return output


# ============================================================
# 保留原有的 Approaches（向后兼容）
# ============================================================

class BaselineApproach(BaseApproach):
    """
    The basic zero-shot CoT approach.
    """

    def solve(self, item: AERItem, prompt_name: str = "cot") -> str:
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
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
        
        # 新增：后处理
        raw_answers = self._parse_answer_from_response(response)
        processed_answers = post_process_answers(raw_answers, item.options)
        
        if processed_answers != raw_answers:
            response += f"\n\n[Post-processing: {raw_answers} -> {processed_answers}]"
        
        return response


class SelfConsistencyRefinementApproach(BaseApproach):
    """
    Combines Self-Consistency (multiple sampling + voting) with Self-Refinement.
    
    UPDATED: Now uses option-level voting instead of answer-set voting.
    """
    
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 5
        self.temperature = 0.5  # 降低温度
        self.vote_threshold = 3  # 至少 3/5 才选
        self.d_option_threshold = 4  # D选项更严格
    
    def _get_prompt(self, item: AERItem, prompt_name: str) -> tuple:
        """Get the system and user prompts."""
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
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
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        """
        Main solving method with improved option-level voting.
        """
        system_prompt, user_prompt, docs_text, options_text, event = self._get_prompt(item, prompt_name)
        
        # ============ STAGE 1: 选项级投票 ============
        print(f"\n[Self-Consistency] Generating {self.num_samples} samples...")
        
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []
        
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=self.temperature)
            all_responses.append(response)
            
            # 选项级投票
            answers = self._parse_answer_from_response(response)
            for opt in answers:
                option_votes[opt] += 1
            
            print(f"  Sample {i+1}: {sorted(answers) if answers else 'No answer'}")
        
        # 基于阈值选择
        # 投票逻辑改成
        voted_answers = set()
        for opt, count in option_votes.items():
            threshold = self.d_option_threshold if opt == 'D' else self.vote_threshold
            if count >= threshold:
                voted_answers.add(opt)
        # 限制逻辑改成：只有选了4个时才移除
        if len(voted_answers) >= 4:
            min_vote = min(option_votes[opt] for opt in voted_answers)
            min_opts = [opt for opt in voted_answers if option_votes[opt] == min_vote]
            if len(min_opts) == 1:
                voted_answers.discard(min_opts[0])
        
        vote_summary = ", ".join(f"{opt}:{count}" for opt, count in sorted(option_votes.items()))
        print(f"\n[Vote counts] {vote_summary}")
        #print(f"[Threshold {self.vote_threshold}] Selected: {sorted(voted_answers)}")
        print(f"[Threshold: general={self.vote_threshold}, D={self.d_option_threshold}] Selected: {sorted(voted_answers)}")
        
        # 如果没有超过阈值的，选最高票的
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {opt for opt, count in option_votes.items() 
                               if count == max_votes}
        
        # ============ STAGE 2: 针对性验证（可选） ============
        # 找出"边缘选项"（得票在阈值附近的）
        uncertain_options = {opt for opt, count in option_votes.items() 
                           if 1 < count < self.vote_threshold}
        
        if uncertain_options:
            print(f"\n[Verification] Uncertain options: {sorted(uncertain_options)}")
            # 这里可以添加针对性验证逻辑
        
        # ============ 后处理 ============
        final_answers = post_process_answers(voted_answers, item.options)
        
        # 构建输出
        output = f"""
========== SELF-CONSISTENCY (Option-Level Voting) ==========
Samples: {self.num_samples}, Threshold: {self.vote_threshold}
Vote counts: {vote_summary}
Voted answers: {sorted(voted_answers)}
After post-processing: {sorted(final_answers)}

========== BEST RESPONSE ==========
{all_responses[0] if all_responses else "No response"}

Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
"""
        return output
