import json
from dataclasses import dataclass
from typing import List

@dataclass
class aerItem:
    """
    A customized data type class.
    """
    id: int
    event: str
    documents: List[str]
    options: List[str]
    answer: str

class dataLoader:
    """
    The main class for data loader.
    """
    def __init__(self, docs_path: str, questions_path: str):
        self.docs_path = docs_path
        self.questions_path = questions_path

        # We load docs.json file in advance, cuz it would be used every iteration.
        self.docs_data = self._load_json_data()

    def _load_json_data(self):
        try:
            with open(self.docs_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.docs_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.docs_path}")
            return []

    def load(self):
        # reformat the docs_data into a dict like {<topic_id>: [<corresponding docs content>, ...]}
        docs_dict = {}
        for item in self.docs_data:
            docs_dict[item["topic_id"]] = [doc["content"] for doc in item["docs"]]

        with open(self.questions_path, "r") as f:
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
                    options = [line[f"option_{i}"] for i in ["A", "B", "C", "D"]],
                    answer = line['golden_answer']
                )
                yield aer_item