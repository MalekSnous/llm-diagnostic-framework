"""
RAG (Retrieval-Augmented Generation) improvement strategy.
Implements vector-based retrieval to augment LLM with external knowledge.
"""

import tempfile
import time
from typing import Any, Dict, List, Optional

from ..core.evaluator import EvaluationMetrics
from ..core.llm_client import BaseLLMClient
from ..failure_tests.base_test import TestCase, TestResult

# sentence-transformers and chromadb are imported lazily inside _initialize_rag
# so this module can be imported without the heavy RAG dependencies installed.
from .base_strategy import BaseImprovementStrategy, ImprovementConfig, ImprovementResult


class RAGSystem(BaseImprovementStrategy):
    """
    Retrieval-Augmented Generation system.

    Retrieves relevant context from knowledge base before generation,
    solving knowledge cutoff and domain expertise limitations.
    """

    def __init__(self):
        super().__init__(
            name="RAG System", description="Augment LLM with retrieval from knowledge base"
        )
        self.embedding_model = None
        self.vector_db = None
        self.collection = None

    # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # self.device = 'cpu'

    def configure(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 300,
        **kwargs,
    ) -> ImprovementConfig:
        """
        Configure RAG system.

        Args:
            embedding_model: Sentence transformer model for embeddings
            top_k: Number of documents to retrieve
            chunk_size: Document chunk size in characters
        """
        # Cost: mostly from LLM calls (retrieval is cheap)
        # Assume similar cost to baseline + small overhead
        estimated_cost = 0.10  # Overhead for retrieval

        return ImprovementConfig(
            strategy_name=self.name,
            parameters={
                "embedding_model": embedding_model,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "similarity_threshold": kwargs.get("similarity_threshold", 0.5),
            },
            estimated_cost=estimated_cost,
            estimated_time="Minutes (after setup)",
        )

    def apply(
        self,
        test_cases: List[TestCase],
        baseline_results: List[TestResult],
        config: ImprovementConfig,
        llm_client: BaseLLMClient,
        knowledge_base: Optional[List[str]] = None,
        **kwargs,
    ) -> List[TestResult]:
        """
        Apply RAG to test cases.

        Args:
            test_cases: Test cases
            baseline_results: Baseline results
            config: RAG configuration
            llm_client: LLM client
            knowledge_base: List of documents to index
        """
        start_time = time.time()

        # Step 1: Initialize embedding model and vector DB
        print("Initializing RAG system...")
        self._initialize_rag(config.parameters["embedding_model"])

        # Step 2: Index knowledge base
        if knowledge_base:
            print(f"Indexing {len(knowledge_base)} documents...")
            self._index_documents(knowledge_base, chunk_size=config.parameters["chunk_size"])
        else:
            # Use test case contexts as knowledge base if available
            contexts = [tc.context for tc in test_cases if tc.context]
            if contexts:
                print(f"Using {len(contexts)} test contexts as knowledge base")
                self._index_documents(contexts, chunk_size=config.parameters["chunk_size"])
            else:
                print("Warning: No knowledge base provided")

        # Step 3: Run RAG on test cases
        print("Running RAG-enhanced generation...")
        improved_results = []

        for test_case in test_cases:
            # Retrieve relevant documents
            retrieved_docs = self._retrieve(test_case.input, top_k=config.parameters["top_k"])

            # Build augmented prompt
            augmented_prompt = self._build_rag_prompt(
                query=test_case.input, retrieved_docs=retrieved_docs
            )

            # Generate with augmented prompt
            response = llm_client.generate(
                augmented_prompt, max_tokens=kwargs.get("max_tokens", 200)
            )

            # Evaluate
            success = (
                test_case.expected_output.lower() in response.text.lower()
                if test_case.expected_output
                else True
            )

            improved_results.append(
                TestResult(
                    test_case_id=test_case.id,
                    prediction=response.text,
                    reference=test_case.expected_output,
                    success=success,
                    metrics={
                        "accuracy": 1.0 if success else 0.0,
                        "num_retrieved": len(retrieved_docs),
                    },
                    response=response,
                )
            )

        elapsed = time.time() - start_time

        # Calculate metrics
        baseline_metrics = self._calculate_metrics(baseline_results)
        improved_metrics = self._calculate_metrics(improved_results)

        # Calculate total cost
        total_cost = (
            sum(r.response.cost_usd for r in improved_results if r.response and r.response.cost_usd)
            + config.estimated_cost
        )

        # Store result
        result = ImprovementResult(
            strategy_name=self.name,
            config=config,
            baseline_metrics=baseline_metrics,
            improved_metrics=improved_metrics,
            improvement_delta=self.evaluate_improvement(baseline_metrics, improved_metrics),
            total_cost=total_cost,
            total_time_seconds=elapsed,
            metadata={"num_documents_indexed": self.collection.count() if self.collection else 0},
        )

        self.results.append(result)
        return improved_results

    def _initialize_rag(self, embedding_model_name: str):
        """Initialize embedding model and vector database."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

        chroma_dir = tempfile.mkdtemp(prefix="chromadb_")

        # Initialize ChromaDB (in-memory for simplicity)
        # self.vector_db = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=chroma_dir  # In-memory
        # ))

        self.vector_db = chromadb.PersistentClient(path=chroma_dir)

        # Create collection
        self.collection = self.vector_db.create_collection(
            name="knowledge_base", metadata={"hnsw:space": "cosine"}
        )

    def _index_documents(self, documents: List[str], chunk_size: int = 500):
        """
        Chunk and index documents into vector database.

        Args:
            documents: List of documents to index
            chunk_size: Size of chunks in characters
        """
        # Chunk documents
        chunks = []
        chunk_ids = []

        for doc_idx, doc in enumerate(documents):
            # Simple chunking (character-based)
            for chunk_idx in range(0, len(doc), chunk_size):
                chunk = doc[chunk_idx : chunk_idx + chunk_size]
                if len(chunk) < 50:  # Skip very small chunks
                    continue

                chunks.append(chunk)
                chunk_ids.append(f"doc{doc_idx}_chunk{chunk_idx // chunk_size}")

        if not chunks:
            print("No chunks created from documents")
            return

        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks, show_progress_bar=True, convert_to_numpy=True
        )

        # Add to vector database
        self.collection.add(ids=chunk_ids, embeddings=embeddings.tolist(), documents=chunks)

        print(f"Indexed {len(chunks)} chunks")

    def _retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k most relevant documents for query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved document texts
        """
        if not self.collection or self.collection.count() == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        # Extract documents
        retrieved_docs = results["documents"][0] if results["documents"] else []

        return retrieved_docs

    def _build_rag_prompt(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Build RAG prompt with retrieved context.

        Args:
            query: Original query
            retrieved_docs: Retrieved documents

        Returns:
            Augmented prompt
        """
        if not retrieved_docs:
            return query

        # AJOUTER : Limiter la longueur totale du contexte
        MAX_CONTEXT_TOKENS = 1000  # ~750 mots
        MAX_CHARS_PER_TOKEN = 4  # Approximation : 1 token ≈ 4 chars
        MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * MAX_CHARS_PER_TOKEN  # 4000 chars

        # Tronquer les documents si nécessaire
        truncated_docs = []
        total_chars = 0

        for doc in retrieved_docs:
            if total_chars + len(doc) > MAX_CONTEXT_CHARS:
                # Prendre seulement ce qui reste
                remaining = MAX_CONTEXT_CHARS - total_chars
                if remaining > 100:  # Au moins 100 chars
                    truncated_docs.append(doc[:remaining] + "...")
                break
            else:
                truncated_docs.append(doc)
                total_chars += len(doc)

        # Format retrieved context
        context = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(truncated_docs)])

        # Build RAG prompt (plus concis)
        prompt = f"""Context: {context}

    Question: {query}

    Answer:"""

        return prompt

    #     def _build_rag_prompt(self, query: str, retrieved_docs: List[str]) -> str:
    #         """
    #         Build RAG prompt with retrieved context.

    #         Args:
    #             query: Original query
    #             retrieved_docs: Retrieved documents

    #         Returns:
    #             Augmented prompt
    #         """
    #         if not retrieved_docs:
    #             return query

    #         # Format retrieved context
    #         context = "\n\n".join([
    #             f"[Document {i+1}]\n{doc}"
    #             for i, doc in enumerate(retrieved_docs)
    #         ])

    #         # Build RAG prompt
    #         prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

    # Context:
    # {context}

    # Question: {query}

    # Answer:"""

    #         return prompt

    def _calculate_metrics(self, results: List[TestResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics."""
        metrics = EvaluationMetrics()

        if not results:
            return metrics

        accuracy = sum(r.metrics.get("accuracy", 0) for r in results) / len(results)
        metrics.add_metric("accuracy", accuracy)

        # Average number of documents retrieved
        avg_retrieved = sum(r.metrics.get("num_retrieved", 0) for r in results) / len(results)
        metrics.add_metric("avg_retrieved", avg_retrieved)

        return metrics

    def get_implementation_guide(self) -> Dict[str, Any]:
        """Implementation guide for RAG."""
        return {
            "overview": "Augment LLM with retrieval from external knowledge base",
            "when_to_use": [
                "Knowledge cutoff is a problem (recent events)",
                "Need domain-specific knowledge (medical, legal, etc.)",
                "Have large document corpus to leverage",
                "Information changes frequently (can update KB without retraining)",
                "Want to reduce hallucinations (grounded in retrieved facts)",
            ],
            "architecture": {
                "components": [
                    "1. Document ingestion & chunking",
                    "2. Embedding generation (Sentence Transformers)",
                    "3. Vector database (ChromaDB/Pinecone/Weaviate)",
                    "4. Retrieval (semantic search)",
                    "5. Prompt augmentation",
                    "6. LLM generation",
                ]
            },
            "requirements": {
                "data": "Document corpus (100s to millions of docs)",
                "compute": "Embedding generation (CPU/GPU)",
                "storage": "Vector database (~4KB per document)",
                "cost": "$0-100/month depending on scale",
            },
            "expected_improvements": {
                "knowledge_accuracy": "30-60% improvement on domain-specific tasks",
                "hallucination_reduction": "40-70% fewer fabrications",
                "cost": "Similar to baseline (slightly higher due to longer prompts)",
            },
            "best_practices": [
                "Chunk documents intelligently (semantic boundaries, not fixed size)",
                "Use metadata filtering (date, category, etc.)",
                "Tune top_k (3-5 usually works well)",
                "Add source citations in output",
                "Monitor retrieval quality (are right docs being retrieved?)",
                "Use reranking for better precision",
            ],
            "pitfalls": [
                "Retrieval quality matters: garbage in, garbage out",
                "Longer prompts = higher cost",
                "Latency: retrieval adds 50-200ms",
                "Context window limits: can't retrieve too much",
                "Outdated KB leads to wrong answers",
            ],
            "code_example": """ 
            
from llm_diagnostic.improvements import RAGSystem

# Prepare knowledge base
knowledge_base = [
    "Document 1: Medical facts...",
    "Document 2: More information...",
    # ... more documents
]

# Configure
strategy = RAGSystem()
config = strategy.configure(
    embedding_model="all-MiniLM-L6-v2",
    top_k=3,
    chunk_size=500
)

# Apply
improved_results = strategy.apply(
    test_cases=test_cases,
    baseline_results=baseline_results,
    config=config,
    llm_client=client,
    knowledge_base=knowledge_base
)
""",
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client
    from ..failure_tests.base_test import TestCase

    # Example knowledge base
    knowledge_base = [
        "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
        "It was designed by Gustave Eiffel and stands 324 meters tall.",
        "Initially, it was meant to be temporary but became a permanent landmark.",
    ]

    # Test cases
    test_cases = [
        TestCase(id="test1", input="When was the Eiffel Tower built?", expected_output="1889")
    ]

    strategy = RAGSystem()
    config = strategy.configure()

    client = get_llm_client("gpt-4-turbo-preview")

    # Would run: strategy.apply(test_cases, [], config, client, knowledge_base)
    print("RAG system ready")
    print(strategy.get_implementation_guide())
