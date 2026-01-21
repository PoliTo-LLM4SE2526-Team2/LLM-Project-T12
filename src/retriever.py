from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class DocumentRetriever:
    """
    Retrieves and ranks documents based on relevance to the event.
    Combines BM25 and vector-based retrieval using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, top_k: int = 10,
                 coarse_top_k: int = 30,
                 use_full_content: bool = False,
                 use_gpu: bool = False,
                 rrf_k: int = 60,
                 use_per_option: bool = False,
                 use_reranker: bool = True,
                 reranker_model: str = 'BAAI/bge-reranker-base'):

        self.top_k = top_k
        self.coarse_top_k = coarse_top_k
        self.use_full_content = use_full_content
        self.use_gpu = use_gpu
        self.rrf_k = rrf_k
        self.use_per_option = use_per_option
        self.use_reranker = use_reranker
        
        # Initialize the embedding model (BGE-base: balanced performance and memory usage)
        try:
            model_name = 'BAAI/bge-base-en-v1.5'
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
        
        # Initialize the reranker
        if self.use_reranker:
            try:
                # Try GPU first, fallback to CPU if CUDA is not available
                device = 'cuda' if use_gpu else 'cpu'
                self.reranker = CrossEncoder(reranker_model, device=device)
                print(f"Loaded reranker: {reranker_model} (device: {device})")
            except Exception as e:
                # If GPU fails, try CPU
                if use_gpu:
                    try:
                        print(f"Warning: Failed to load reranker on GPU ({e}), trying CPU...")
                        self.reranker = CrossEncoder(reranker_model, device='cpu')
                        print(f"Loaded reranker: {reranker_model} (device: cpu)")
                    except Exception as e2:
                        print(f"Warning: Failed to load reranker on CPU ({e2}), disabling reranking")
                        self.reranker = None
                else:
                    print(f"Warning: Failed to load reranker ({e}), disabling reranking")
                    self.reranker = None
        else:
            self.reranker = None


    
    def _retrieve_bm25(self, query: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Retrieve all documents using BM25 with scores.
        Returns list of documents sorted by score descending.
        """    
        try:
            # Preprocessing document/snippet
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            tokenized_texts = [item.lower().split(" ") for item in texts_to_index]
            tokenized_query = query.lower().split(" ")
            
            # Retrieve
            bm25 = BM25Okapi(tokenized_texts)
            scores = bm25.get_scores(tokenized_query)
            sorted_indices = np.argsort(scores)[::-1]
            
            results = [documents[i] for i in sorted_indices]
            return results
           
        except Exception as e:
            print(f"Warning: BM25 retrieval failed ({e}).")
            return None
        

    def _retrieve_semantic(self, query: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Semantic retriever using vector embeddings (sentence transformers).
        Returns list of documents sorted by similarity descending.
        """
        try:
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # Encode query and documents
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            doc_embeddings = self.model.encode(
                texts_to_index,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True,
            )

            # Calculate cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

            # Sort all documents by similarity descending
            sorted_indices = np.argsort(similarities)[::-1]

            results = [documents[i] for i in sorted_indices]
            return results
            
        except Exception as e:
            print(f"Warning: Vector retrieval failed ({e})")
            return None
        
    
    def _rrf_merge(self, bm25_results: List[str], 
                   vector_results: List[str]) -> List[Tuple[str, float]]:
        """
        Merge results from BM25 and vector retrieval using Reciprocal Rank Fusion (RRF).
        RRF score: 1 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)

        # Process vector results
        for rank, doc in enumerate(vector_results, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending
        merged_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged_results[:self.coarse_top_k]
    

    def retrieve(self, event: str, title_snippet: List[str], documents: List[str], options: List[str] = None) -> List[str]:
        """
        Two-stage retrieval:
        1. Coarse retrieval (BM25 + Semantic + RRF) → top coarse_top_k candidates
        2. Fine-grained reranking (Cross-Encoder) → top top_k results
        """
        # 如果启用 per-option 且提供了 options，使用新方法
        if self.use_per_option and options:
            return self.retrieve_with_options(event, options, title_snippet, documents)
        
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
    
        # Stage 1: Coarse retrieval (BM25 + Semantic + RRF)
        bm25_results = self._retrieve_bm25(event, title_snippet, documents)
        vector_results = self._retrieve_semantic(event, title_snippet, documents)

        if not vector_results:
            if not bm25_results:
                candidates = documents[:self.coarse_top_k]
            else:
                candidates = [doc for doc in bm25_results[:self.coarse_top_k]]
        else:
            # Merge using RRF
            merged_results = self._rrf_merge(bm25_results, vector_results)
            candidates = [doc for doc, _ in merged_results]
        
        # Stage 2: Reranking (if enabled and necessary)
        if self.reranker and len(candidates) > self.top_k:
            candidates = self._rerank(event, candidates)
        
        return candidates[:self.top_k]
    

    def retrieve_with_options(self, event: str, options: List[str],
                              title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Two-stage retrieval with per-option weighting:
        Stage 1: Event (2x weight) + Options (1x weight) → coarse_top_k candidates
        Stage 2: Rerank with event + all options combined → top_k results
        """
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
        
        # Stage 1: Coarse retrieval with weighted RRF
        all_scores: Dict[str, float] = {}
        
        # 1. Event 相关（权重 2x）
        bm25_event = self._retrieve_bm25(event, title_snippet, documents)
        vec_event = self._retrieve_semantic(event, title_snippet, documents)
        
        if bm25_event:
            for rank, doc in enumerate(bm25_event, 1):
                all_scores[doc] = all_scores.get(doc, 0) + 2.0 / (self.rrf_k + rank)
        if vec_event:
            for rank, doc in enumerate(vec_event, 1):
                all_scores[doc] = all_scores.get(doc, 0) + 2.0 / (self.rrf_k + rank)
        
        # 2. 每个 option 相关（权重 1x，BM25 + Semantic）
        for option in options:
            bm25_opt = self._retrieve_bm25(option, title_snippet, documents)
            vec_opt = self._retrieve_semantic(option, title_snippet, documents)
            
            if bm25_opt:
                for rank, doc in enumerate(bm25_opt, 1):
                    all_scores[doc] = all_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)
            if vec_opt:
                for rank, doc in enumerate(vec_opt, 1):
                    all_scores[doc] = all_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)
        
        # Get top coarse_top_k candidates
        sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc for doc, _ in sorted_docs[:self.coarse_top_k]]
        
        # Stage 2: Reranking with combined query (event + all options)
        if self.reranker and len(candidates) > self.top_k:
            combined_query = f"{event} {' '.join(options)}"
            candidates = self._rerank(combined_query, candidates)
        
        return candidates[:self.top_k]
    
    
    def _rerank(self, query: str, candidates: List[str]) -> List[str]:
        """
        Rerank candidates using Cross-Encoder for more accurate scoring.
        Returns candidates sorted by relevance score descending.
        """
        try:
            # Build query-document pairs
            pairs = [[query, doc] for doc in candidates]
            
            # Batch prediction for efficiency
            scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Sort by score descending
            sorted_indices = np.argsort(scores)[::-1]
            return [candidates[i] for i in sorted_indices]
        
        except Exception as e:
            print(f"Warning: Reranking failed ({e}), returning original order")
            return candidates