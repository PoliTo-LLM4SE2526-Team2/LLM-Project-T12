import json
from typing import List

class aerItem:
    """
    A data type class
    """
    id: int
    event: str
    documents: List[str]
    options: List[str]
    answer: str = None

class dataLoader:
    """
    The main class for data loader.
    """
    def __init__(self, json_path: str, jsonl_path: str):
        self.json_path = json_path
        self.jsonl_path = jsonl_path

    def load(self):
        with open(self.josn_path, 'r') as f:
            docs_data = json.load(f)
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.scrip():
                    aer_item = aerItem(
                        id = line["topic_id"],
                        event = line["target_event"],
                        documents = [d["content"] for i in docs_data if i["topic_id"] == id for d in i["docs"]],
                        options = [line[f"option{i}"] for i in ["A", "B", "C", "D"]],
                        answer = line['golden_answer']
                    )
                    yield aer_item