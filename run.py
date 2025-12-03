import argparse
from src.dataloader import DataLoader
from src.llm import ChatLLM
from src.approaches import BaselineApproach

MODEL_NAME = "deepseek-reasoner"
API_KEY = "sk-6d964559089a47c488dde2fcb7e3b6fe"
BASE_URL = "https://api.deepseek.com"

def main():
    # to define parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str, default="data/dev/docs.json")
    parser.add_argument("--questions_path", type=str, default="data/dev/questions.jsonl")
    parser.add_argument("--limit", type=int, default=2, help="Run the first 2 events to test.")
    args = parser.parse_args()

    # initialize components
    # we use Deepseek api right now, just modify parameters of chatLLM() if we want ChatGPT instead
    # we firstly use the baseline model to solve, for details please refer to "src/solvers.py"
    llm = ChatLLM(model_name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL)
    solver = BaselineApproach(llm) # change this component if we want to use another solve method
    loader = DataLoader(args.docs_path, args.questions_path)

    print(f"Running experiment with {solver.__class__.__name__}...\n")

    correct_count = 0
    bad_cases = {} # include all uuid of incorrectly predicted events and llm reasoning process
    for i, event in enumerate(loader.load()):
        if i <= args.limit - 1:
            print(f"--- Processing Item {i+1} ---")
            prediction = solver.solve(event)

            print("-" * 30)
            print(prediction)
            print(f"\nGround truth: {event.answer}")
            print("-" * 30 + "\n")

            if sorted(prediction.split("Final Answer I Reasoned: ")[-1].split(",")) == sorted(event.answer.split(",")):
                correct_count += 1
            else:
                bad_cases[event.event_uuid] = prediction
        else:
            break
    
    print(f"Accuracy: {correct_count / args.limit * 100:.2f}%\n")

if __name__ == "__main__":
    main()



