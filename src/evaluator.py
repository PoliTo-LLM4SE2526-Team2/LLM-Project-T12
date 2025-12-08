from typing import Dict, List, Set
from collections import defaultdict
import json


class Evaluator:
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.incorrect = 0
        
        # For precision/recall calculation (treating as multi-label classification)
        self.true_positives = defaultdict(int)  # option -> count
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        
        # Option-level statistics
        self.option_stats = defaultdict(lambda: {"correct": 0, "total_selected": 0, "total_should_select": 0})
        
        # Error cases
        self.error_cases = []
        
        # Answer type statistics
        self.single_answer = defaultdict(int)
        self.multi_answer = defaultdict(int)
        self.insufficient_info_count = 0
        self.insufficient_info_correct = 0
    
    def update(self, predicted: Set[str], ground_truth: Set[str], event_uuid: str = "", 
               prediction_text: str = "", event: str = "", options: List[str] = None):
       
        self.total += 1
        
        # Check if completely correct
        is_correct = (predicted == ground_truth)
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1
            # Store error case
            if event_uuid:
                self.error_cases.append({
                    "uuid": event_uuid,
                    "event": event,
                    "predicted": sorted(list(predicted)),
                    "ground_truth": sorted(list(ground_truth)),
                    "prediction_text": prediction_text,
                    "options": [f"option_{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], options)]
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
        stats["correct"] += is_correct
        
        # Check for "insufficient information" (usually option C with specific text)
        if options:
            for i, opt in enumerate(options):
                if "insufficient" in opt.lower() or "none of" in opt.lower():
                    option_label = ["A", "B", "C", "D"][i]
                    self.insufficient_info_count += 1
                    if option_label in ground_truth and option_label in predicted:
                        self.insufficient_info_correct += 1
                    break
    
    def get_accuracy(self) -> float:
        """Calculate overall accuracy."""
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
        """
        {
        "A": {"precision": <float>, "recall": <float>, "f1": <float>},
        "B": {...},
        "C": {...},
        "D": {...}
        }
        correct: tp
        total_selected: tp+fp
        total_should_select: tp+fn
        """
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
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.get_accuracy(),
            "macro_f1": self.get_macro_f1(),
            "insufficient_info_count": self.insufficient_info_count,
            "insufficient_info_accuracy": self.get_insufficient_info_accuracy(),
            "single_answer_count": self.single_answer["count"],
            "single_answer_accuracy": self.get_single_answer_accuracy(),
            "multi_answer_count": self.multi_answer["count"],
            "multi_answer_accuracy": self.get_multi_answer_accuracy(),
            "option_stats": dict(self.option_stats),
            "option_matrix": self.get_option_matrix(),
            "error_count": len(self.error_cases)
        }
    
    def save_results(self, filepath: str, approach_name: str = "BaselineApproach"):
        """Save evaluation results to JSON file."""
        results = {
            "approach": approach_name,
            "summary": self.get_summary(),
            "error_cases": self.error_cases
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")

