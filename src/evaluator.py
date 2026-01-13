"""
Evaluator for SemEval 2026 Task 12: Abductive Event Reasoning

Official Evaluation Metric:
- 1.0 (Full Match): P = G
- 0.5 (Partial Match): P is a non-empty proper subset of G (no wrong selections)
- 0.0 (Incorrect): otherwise (any wrong selection or empty)

Final score = average score across all instances
"""

from typing import Dict, List, Set
from collections import defaultdict
import json


class Evaluator:
    
    def __init__(self):
        self.total = 0
        self.correct = 0  # Full match (1.0)
        self.partial = 0  # Partial match (0.5)
        self.incorrect = 0  # Incorrect (0.0)
        
        # For precision/recall calculation (treating as multi-label classification)
        self.true_positives = defaultdict(int)  # option -> count
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        
        # Option-level statistics
        self.option_stats = defaultdict(lambda: {"correct": 0, "total_selected": 0, "total_should_select": 0})
        
        # Error cases (分类存储)
        self.error_cases = []
        self.partial_cases = []  # 新增：partial match 的案例
        
        # Answer type statistics
        self.single_answer = defaultdict(int)
        self.multi_answer = defaultdict(int)
        self.insufficient_info_count = 0
        self.insufficient_info_correct = 0
        
        # 新增：详细的错误分类统计
        self.error_types = {
            "over_selection": 0,    # 多选了错误选项
            "under_selection": 0,   # 少选了正确选项（但没选错）-> 这是 partial
            "wrong_selection": 0,   # 选错了（有 false positive）
            "empty_prediction": 0,  # 空预测
        }
    
    def _calculate_instance_score(self, predicted: Set[str], ground_truth: Set[str]) -> float:
        """
        Calculate score for a single instance based on official metric.
        
        Returns:
            1.0: Full match (P = G)
            0.5: Partial match (P ⊂ G, P ≠ ∅)
            0.0: Incorrect
        """
        if not predicted:  # Empty prediction
            return 0.0
        
        if predicted == ground_truth:  # Perfect match
            return 1.0
        
        # Check if P is a proper subset of G (partial match)
        # P ⊂ G means: P is subset of G AND P ≠ G
        # Also requires: no false positives (all predicted are in ground truth)
        if predicted < ground_truth:  # proper subset
            # predicted is non-empty (checked above) and all predictions are correct
            return 0.5
        
        # Any other case: wrong selection or superset
        return 0.0
    
    def _classify_error(self, predicted: Set[str], ground_truth: Set[str]) -> str:
        """Classify the type of error."""
        if not predicted:
            return "empty_prediction"
        
        false_positives = predicted - ground_truth  # 选错的
        false_negatives = ground_truth - predicted  # 漏选的
        
        if false_positives:
            return "wrong_selection"  # 有错选
        elif false_negatives:
            return "under_selection"  # 只有漏选（这其实是 partial match）
        else:
            return "unknown"
    
    def update(self, predicted: Set[str], ground_truth: Set[str], event_id: str = "", 
               prediction_text: str = "", event: str = "", options: List[str] = None):
        
        self.total += 1
        
        # Calculate instance score using official metric
        score = self._calculate_instance_score(predicted, ground_truth)
        
        if score == 1.0:
            self.correct += 1
        elif score == 0.5:
            self.partial += 1
            # 记录 partial match 案例
            if event_id:
                self.partial_cases.append({
                    "id": event_id,
                    "event": event,
                    "predicted": sorted(list(predicted)),
                    "ground_truth": sorted(list(ground_truth)),
                    "missing": sorted(list(ground_truth - predicted)),
                    "score": 0.5
                })
        else:
            self.incorrect += 1
            # 分类错误类型
            error_type = self._classify_error(predicted, ground_truth)
            self.error_types[error_type] += 1
            
            # Store error case
            if event_id:
                false_positives = predicted - ground_truth
                false_negatives = ground_truth - predicted
                self.error_cases.append({
                    "id": event_id,
                    "event": event,
                    "predicted": sorted(list(predicted)),
                    "ground_truth": sorted(list(ground_truth)),
                    "false_positives": sorted(list(false_positives)),
                    "false_negatives": sorted(list(false_negatives)),
                    "error_type": error_type,
                    "prediction_text": prediction_text,
                    "options": [f"option_{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], options)] if options else []
                })
        
        # Update option-level statistics
        all_options = predicted | ground_truth
        for option in all_options:
            if option in predicted and option in ground_truth:
                self.true_positives[option] += 1
                self.option_stats[option]["correct"] += 1
            elif option in predicted and option not in ground_truth:
                self.false_positives[option] += 1
            elif option not in predicted and option in ground_truth:
                self.false_negatives[option] += 1
            
            if option in predicted:
                self.option_stats[option]["total_selected"] += 1
            if option in ground_truth:
                self.option_stats[option]["total_should_select"] += 1
        
        # Answer type statistics
        is_single = (len(ground_truth) == 1)
        stats = self.single_answer if is_single else self.multi_answer
        stats["count"] += 1
        stats["correct"] += (score == 1.0)
        stats["partial"] = stats.get("partial", 0) + (score == 0.5)
        
        # Check for "insufficient information" (usually option with "none of" text)
        if options:
            for i, opt in enumerate(options):
                if "insufficient" in opt.lower() or "none of" in opt.lower():
                    option_label = ["A", "B", "C", "D"][i]
                    self.insufficient_info_count += 1
                    if option_label in ground_truth and option_label in predicted:
                        self.insufficient_info_correct += 1
                    break
    
    def get_official_score(self) -> float:
        """
        Calculate the official SemEval score.
        
        Score = (1.0 * full_match + 0.5 * partial_match + 0.0 * incorrect) / total
        """
        if self.total == 0:
            return 0.0
        return (1.0 * self.correct + 0.5 * self.partial) / self.total
    
    def get_accuracy(self) -> float:
        """Calculate strict accuracy (full match only)."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def get_macro_f1(self) -> float:
        """Calculate macro-averaged F1 score."""
        all_options = set(self.true_positives.keys()) | set(self.false_positives.keys()) | set(self.false_negatives.keys())
        
        if not all_options:
            return 0.0
        
        f1_scores = []
        for option in all_options:
            tp = self.true_positives[option]
            fp = self.false_positives[option]
            fn = self.false_negatives[option]

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)
    
    def get_insufficient_info_accuracy(self) -> float:
        """Calculate accuracy for 'insufficient information' cases."""
        if self.insufficient_info_count == 0:
            return 0.0
        return self.insufficient_info_correct / self.insufficient_info_count
    
    def get_single_answer_accuracy(self) -> float:
        """Calculate accuracy for 'single answer' cases."""
        if self.single_answer["count"] == 0:
            return 0.0
        return self.single_answer["correct"] / self.single_answer["count"]
    
    def get_multi_answer_accuracy(self) -> float:
        """Calculate accuracy for 'multi answer' cases."""
        if self.multi_answer["count"] == 0:
            return 0.0
        return self.multi_answer["correct"] / self.multi_answer["count"]
    
    def get_option_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get precision/recall/f1 for each option."""
        option_matrix = {}
        for option, stats in self.option_stats.items():
            option_info = defaultdict(float)
            prec = (stats["correct"] / stats["total_selected"]) if stats["total_selected"] > 0 else 0.0
            rec = (stats["correct"] / stats["total_should_select"]) if stats["total_should_select"] > 0 else 0.0
            f1 = 2 * ((prec * rec) / (prec + rec)) if prec + rec > 0 else 0.0
            
            option_info["precision"] = prec
            option_info["recall"] = rec
            option_info["f1"] = f1

            option_matrix[option] = option_info

        return option_matrix
    
    def get_summary(self) -> Dict:
        """Get comprehensive evaluation summary."""
        return {
            "total": self.total,
            # 官方指标
            "official_score": self.get_official_score(),
            "full_match": self.correct,
            "partial_match": self.partial,
            "incorrect": self.incorrect,
            # 传统指标
            "strict_accuracy": self.get_accuracy(),
            "macro_f1": self.get_macro_f1(),
            # 错误分类
            "error_types": self.error_types,
            # 分类统计
            "insufficient_info_count": self.insufficient_info_count,
            "insufficient_info_accuracy": self.get_insufficient_info_accuracy(),
            "single_answer_count": self.single_answer["count"],
            "single_answer_accuracy": self.get_single_answer_accuracy(),
            "single_answer_partial": self.single_answer.get("partial", 0),
            "multi_answer_count": self.multi_answer["count"],
            "multi_answer_accuracy": self.get_multi_answer_accuracy(),
            "multi_answer_partial": self.multi_answer.get("partial", 0),
            # 选项级统计
            "option_stats": dict(self.option_stats),
            "option_matrix": self.get_option_matrix(),
            "error_count": len(self.error_cases),
            "partial_count": len(self.partial_cases)
        }
    
    def print_summary(self):
        """Print a formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY (SemEval 2026 Task 12 Metric)")
        print("=" * 60)
        
        print(f"\n📊 OFFICIAL SCORE: {summary['official_score']:.4f}")
        print(f"   (1.0×{summary['full_match']} + 0.5×{summary['partial_match']} + 0.0×{summary['incorrect']}) / {summary['total']}")
        
        print(f"\n📈 Score Breakdown:")
        print(f"   Full Match (1.0):    {summary['full_match']:4d} ({summary['full_match']/summary['total']*100:.1f}%)")
        print(f"   Partial Match (0.5): {summary['partial_match']:4d} ({summary['partial_match']/summary['total']*100:.1f}%)")
        print(f"   Incorrect (0.0):     {summary['incorrect']:4d} ({summary['incorrect']/summary['total']*100:.1f}%)")
        
        print(f"\n❌ Error Types:")
        for error_type, count in summary['error_types'].items():
            if count > 0:
                print(f"   {error_type}: {count}")
        
        print(f"\n📋 By Answer Type:")
        print(f"   Single-answer: {summary['single_answer_accuracy']:.2%} accuracy, {summary['single_answer_count']} cases")
        print(f"   Multi-answer:  {summary['multi_answer_accuracy']:.2%} accuracy, {summary['multi_answer_count']} cases")
        
        print(f"\n📊 Option-Level Performance:")
        print(f"   {'Option':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        for option, matrix in sorted(summary['option_matrix'].items()):
            print(f"   {option:<8} {matrix['precision']:.4f}       {matrix['recall']:.4f}       {matrix['f1']:.4f}")
        
        print("=" * 60)
    
    def save_results(self, filepath: str, approach_name: str = "BaselineApproach"):
        """Save evaluation results to JSON file."""
        results = {
            "approach": approach_name,
            "summary": self.get_summary(),
            "error_cases": self.error_cases,
            "partial_cases": self.partial_cases  # 新增
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
