from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. SemanticRetriever will not work.")
    print("Install it with: pip install sentence-transformers")


class DocumentRetriever:
    """
    Retrieves and ranks documents based on relevance to the event.
    Uses BM25 (Best Matching 25)
    """
    
    def __init__(self, top_k: int = 10, use_full_content: bool = False):
        """
        Initialize BM25 retriever.
        
        Args:
            top_k: Number of top documents to retrieve
            use_full_content: If True, use full document content for indexing.
                             If False, use title_snippet (default, for backward compatibility)
        """
        self.top_k = top_k
        self.use_full_content = use_full_content
        # BM25 is instantiated per query set in this design
    
    def retrieve(self, event: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Retrieve documents based on BM25 relevance to the event.
        
        Args:
            event: The event description to search for
            title_snippet: List of title+snippet strings (used if use_full_content=False)
            documents: List of full document content strings
            
        Returns:
            List of retrieved documents (top_k most relevant)
        """
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
        
        try:
            # 1. Choose which text to use for indexing
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # 2. Preprocessing: Tokenization
            tokenized_texts = [item.lower().split(" ") for item in texts_to_index]
            tokenized_event = event.lower().split(" ")
            
            # 3. Build Index
            bm25 = BM25Okapi(tokenized_texts)
            
            # 4. Retrieve Top-N
            # Note: get_top_n returns documents in the order of the original documents list
            # We need to map back to the correct documents
            if self.use_full_content:
                # If using full content, directly retrieve from documents
                retrieved_docs = bm25.get_top_n(tokenized_event, documents, n=self.top_k)
            else:
                # If using title_snippet, we need to map back to full documents
                # Get scores for all documents
                scores = bm25.get_scores(tokenized_event)
                # Get top_k indices
                top_indices = np.argsort(scores)[::-1][:self.top_k]
                # Map back to full documents
                retrieved_docs = [documents[i] for i in top_indices]
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Warning: BM25 retrieval failed ({e}), using first {self.top_k} documents")
            return documents[:self.top_k]

    
    def retrieve_with_scores(self, event: str, title_snippet: List[str], documents: List[str]) -> List[Tuple[str, float]]:
        """
        Return documents with their BM25 scores.
        
        Args:
            event: The event description to search for
            title_snippet: List of title+snippet strings (used if use_full_content=False)
            documents: List of full document content strings
            
        Returns:
            List of tuples: [(document, bm25_score), ...]
        """
        if not documents:
            return []
        
        try:
            # Choose which text to use for indexing
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # Preprocessing: Tokenization
            tokenized_texts = [item.lower().split(" ") for item in texts_to_index]
            tokenized_event = event.lower().split(" ")
            
            # Build Index
            bm25 = BM25Okapi(tokenized_texts)
            
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


class SemanticRetriever:
    """
    Semantic retriever using vector embeddings (sentence transformers).
    Retrieves documents based on semantic similarity rather than keyword matching.
    """
    
    def __init__(self, top_k: int = 10, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_full_content: bool = True,
                 use_gpu: bool = False):
        """
        Initialize semantic retriever.
        
        Args:
            top_k: Number of top documents to retrieve
            model_name: Name of the sentence transformer model
                - 'all-MiniLM-L6-v2': Fast, 384-dim (default)
                - 'all-mpnet-base-v2': Slower but more accurate, 768-dim
                - 'all-MiniLM-L12-v2': Balanced, 384-dim
            use_full_content: If True, use full document content for indexing.
                             If False, use title_snippet (for compatibility with BM25)
            use_gpu: Whether to use GPU for encoding (if available)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SemanticRetriever. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.top_k = top_k
        self.model_name = model_name
        self.use_full_content = use_full_content
        self.use_gpu = use_gpu
        
        # Initialize the model
        try:
            self.model = SentenceTransformer(model_name)
            if use_gpu:
                try:
                    self.model = self.model.to('cuda')
                    print(f"Using GPU for semantic retrieval")
                except:
                    print(f"GPU not available, using CPU")
            print(f"Loaded semantic retrieval model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def retrieve(self, event: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Retrieve documents based on semantic similarity to the event.
        
        Args:
            event: The event description to search for
            title_snippet: List of title+snippet strings (used if use_full_content=False)
            documents: List of full document content strings
            
        Returns:
            List of retrieved documents (top_k most similar)
        """
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
        
        try:
            # Choose which text to use for indexing
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # Encode event and documents
            event_embedding = self.model.encode(
                [event],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            doc_embeddings = self.model.encode(
                texts_to_index,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,  # Batch processing for efficiency
                normalize_embeddings=True
            )
            
            # Calculate cosine similarity (dot product since vectors are normalized)
            # event_embedding shape: (1, dim)
            # doc_embeddings shape: (n_docs, dim)
            similarities = np.dot(doc_embeddings, event_embedding.T).flatten()
            
            # Get top_k indices (sorted by similarity, descending)
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            # Return full document content (even if indexed with title_snippet)
            return [documents[i] for i in top_indices]
            
        except Exception as e:
            print(f"Warning: Semantic retrieval failed ({e}), using first {self.top_k} documents")
            return documents[:self.top_k]
    
    def retrieve_with_scores(self, event: str, title_snippet: List[str], 
                            documents: List[str]) -> List[Tuple[str, float]]:
        """
        Retrieve documents with their similarity scores.
        
        Returns:
            List of tuples: [(document, similarity_score), ...]
            Scores range from -1 to 1 (cosine similarity)
        """
        if not documents:
            return []
        
        try:
            # Choose which text to use for indexing
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # Encode
            event_embedding = self.model.encode(
                [event],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            doc_embeddings = self.model.encode(
                texts_to_index,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
            
            # Calculate similarities
            similarities = np.dot(doc_embeddings, event_embedding.T).flatten()
            
            # Sort and get top_k
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            results = []
            for i in top_indices:
                results.append((documents[i], float(similarities[i])))
            
            return results
            
        except Exception as e:
            print(f"Warning: Semantic retrieval with scores failed ({e})")
            return [(doc, 0.0) for doc in documents[:self.top_k]]