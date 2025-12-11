from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

class DocumentRetriever:
    """
    Retrieves and ranks documents based on relevance to the event.
    Uses BM25 (Best Matching 25)
    """
    
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        # BM25 is instantiated per query set in this design
    
    def retrieve(self, event: str, documents: List[str]) -> List[str]:
       
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
        
        try:
            # 1. Preprocessing: Tokenization
            tokenized_docs = [doc.lower().split(" ") for doc in documents]
            tokenized_event = event.lower().split(" ")
            
            # 2. Build Index
            bm25 = BM25Okapi(tokenized_docs)
            
            # 3. Retrieve Top-N
            retrieved_docs = bm25.get_top_n(tokenized_event, documents, n=self.top_k)
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Warning: BM25 retrieval failed ({e}), using first {self.top_k} documents")
            return documents[:self.top_k]

    
    def retrieve_with_scores(self, event: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Return documents with their BM25 scores.
        """
        if not documents:
            return []
        
        try:
            tokenized_docs = [doc.lower().split(" ") for doc in documents]
            tokenized_event = event.lower().split(" ")
            
            bm25 = BM25Okapi(tokenized_docs)
            
            # Calculate scores for all documents
            scores = bm25.get_scores(tokenized_event)
            
            # Sort manually to get top_k
            # argsort returns lowest to highest, so we reverse it [::-1]
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            
            results = []
            for i in top_indices:
                results.append((documents[i], float(scores[i])))
                
            return results
            
        except Exception as e:
            print(f"Warning: BM25 score retrieval failed ({e})")
            return [(doc, 0.0) for doc in documents[:self.top_k]]