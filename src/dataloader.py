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
        self.docs_data = self._load_json_data()

    def _load_json_data(self):
        try:
            with open(self.json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.json_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.json_path}")
            return []

    def load(self):
        # reformat the docs_data into a dict like {<topic_id>: [<corresponding docs content>, ...]}
        docs_dict = {}
        for item in self.docs_data:
            docs_dict[item["topic_id"]] = [doc["content"] for doc in item["docs"]]

        with open(self.jsonl_path, "r") as f:
            for line_str in f:
                try:
                    line = json.loads(line_str)
                except json.JSONDecodeError:
                    continue
                topic_id = line["topic_id"]
                documents = docs_dict.get(topic_id, [])
                aer_item = aerItem(
                    id = topic_id,
                    event = line["target_event"],
                    documents = documents,
                    options = [line[f"option{i}"] for i in ["A", "B", "C", "D"]],
                    answer = line['golden_answer']
                )
                yield aer_item