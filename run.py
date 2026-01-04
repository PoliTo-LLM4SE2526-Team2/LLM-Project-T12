import argparse
import concurrent.futures
from datetime import datetime
import os
import re
import json
import time
from src.dataloader import DataLoader
from src.llm import ChatLLM
from src.approaches import BaselineApproach
from src.evaluator import Evaluator
from src.retriever import DocumentRetriever
from dotenv import load_dotenv


def parse_answer(prediction: str) -> set:
   
    if not prediction:
        return set()
    
    # Try to find "Final Answer I Reasoned: ..." pattern
    pattern = r"Final Answer I Reasoned:\s*([A-D,\s]+)"
    match = re.search(pattern, prediction, re.IGNORECASE)
    
    if match:
        answer_str = match.group(1).strip()
        # Split by comma and clean up
        answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
        # Filter valid options (A, B, C, D)
        valid_answers = {a for a in answers if a in ["A", "B", "C", "D"]}
        return valid_answers
    
    # Fallback: try to find any single letter A-D at the end
    pattern2 = r"\b([A-D])\b"
    matches = re.findall(pattern2, prediction[-200:])  # Check last 200 chars
    if matches:
        return {m.upper() for m in matches if m.upper() in ["A", "B", "C", "D"]}
    
    return set()


def parse_ground_truth(answer_str: str) -> set:
    
    if not answer_str:
        return set()
    answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
    return {a for a in answers if a in ["A", "B", "C", "D"]}


def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str, default="data/dev/docs.json")
    parser.add_argument("--questions_path", type=str, default="data/dev/questions.jsonl")
    parser.add_argument("--submission_path", type=str, default="submission.jsonl")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve (0 = use all)")
    parser.add_argument("--no_retrieval", action="store_true", help="Disable document retrieval (use all documents)")
    args = parser.parse_args()

    # initialize components
    load_dotenv()
    llm = ChatLLM(model_name=os.getenv("MODEL_NAME"), api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    retriever = None if args.no_retrieval else DocumentRetriever(top_k=args.top_k if args.top_k > 0 else 1000)

    solver = BaselineApproach(llm, retriever) # change this component if we want to use another solve method
    loader = DataLoader(args.docs_path, args.questions_path)
    evaluator = Evaluator()
    submission = []
    start_time = time.time()

    print(f"Running experiment with {solver.__class__.__name__}...")
    if retriever != None:
        print(f"Document retrieval: Enabled (top_k={args.top_k})\n")
    else:
        print("Document retrieval: Disabled (using all documents)\n")
    print()

    max_workers = int(os.getenv("MAX_WORKERS"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit each event in the dataset to excutor
        # executor.submit(function, arg1, arg2, ...)
        future_to_event = {executor.submit(solver.solve, event): event for event in loader.load()}

        for future in concurrent.futures.as_completed(future_to_event):
            event = future_to_event[future]
            try:
                prediction = future.result()

                # parse answers
                predicted = parse_answer(prediction)
                ground_truth = parse_ground_truth(event.answer)

                # update evaluator
                evaluator.update(
                    predicted=predicted,
                    ground_truth=ground_truth,
                    event_uuid=event.event_uuid,
                    prediction_text=prediction,
                    event=event.event,
                    options=event.options
                )

                # save to submission
                predicted_str = ",".join(sorted(predicted))
                submission_answer = {"id": event.event_uuid, "answer": predicted_str}
                submission.append(submission_answer)


                # Print result with reasoning
                print("-" * 50)
                print("REASONING PROCESS:")
                print("-" * 50)
                # Extract reasoning part (everything before "Final Answer")
                # reasoning_match = re.search(r"(.+?)(?:Final Answer|$)", prediction, re.DOTALL | re.IGNORECASE)
                # if reasoning_match:
                #     reasoning = reasoning_match.group(1).strip()
                #     print(reasoning)
                # else:
                #     print(prediction)
                print(prediction)
                print("-" * 50)
                print(f"Predicted Answer: {sorted(list(predicted))}")
                print(f"Ground Truth: {sorted(list(ground_truth))}")
                print(f"Correct: {predicted == ground_truth}")
                print("-" * 50)
                print()

            except Exception as e:
                print(f'{event.id} generated an exception: {e}')
                evaluator.update(
                    predicted=set(),
                    ground_truth=parse_ground_truth(event.answer),
                    event_uuid=event.event_uuid,
                    prediction_text="",
                    event=event.event,
                    options=event.options
                )

    end_time = time.time()
    total_time = end_time - start_time

    # write to submission.jsonl
    with open(args.submission_path, 'w') as f:
        for item in submission:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # print evaluation summary
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Approach: {solver.__class__.__name__}")
    print(f"LLM Model: {os.getenv("MODEL_NAME")}")
    print(f"Total Time: {total_time:.2f} seconds")
    summary = evaluator.get_summary()
    print(f"\nTotal: {summary['total']}")
    print(f"Correct: {summary['correct']}")
    print(f"Incorrect: {summary['incorrect']}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print(f"Macro F1 Score: {summary['macro_f1']:.4f}")
    print(f"\nSingle Answer Accuracy: {summary['single_answer_accuracy']:.4f} ({summary['single_answer_count']} cases)")
    print(f"Multi Answer Accuracy: {summary['multi_answer_accuracy']:.4f} ({summary['multi_answer_count']} cases)")
    print(f"Insufficient Info Accuracy: {summary['insufficient_info_accuracy']:.4f} ({summary['insufficient_info_count']} cases)")
    print(f"\nOption Level Matrix:")
    print("\tPrecision\tRecall\t\tF1")
    for option, matrix in sorted(summary['option_matrix'].items()):
        print(f"{option}\t{matrix["precision"]:.4f}\t\t{matrix["recall"]:.4f}\t\t{matrix["f1"]:.4f}")
    print("=" * 50)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    evaluator.save_results(output_file, approach_name=solver.__class__.__name__)
    
    print(f"\nEvaluation complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
