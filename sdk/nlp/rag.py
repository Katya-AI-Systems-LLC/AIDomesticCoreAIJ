"""
RAG Pipeline
============

Retrieval Augmented Generation for LLMs.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Retrieved context for RAG."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """RAG response."""
    answer: str
    contexts: List[RAGContext]
    model: str
    latency_ms: float
    tokens_used: int = 0


class RAGPipeline:
    """
    Retrieval Augmented Generation pipeline.
    
    Features:
    - Document retrieval
    - Context injection
    - Multiple LLM backends
    - Reranking
    - Citation generation
    
    Example:
        >>> rag = RAGPipeline(search_engine, llm_client)
        >>> response = rag.query("What is quantum computing?")
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the following context to answer the question. If the context doesn't contain relevant information,
say so clearly. Always cite your sources when possible.

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(self, search_engine: Any = None,
                 llm_client: Any = None,
                 top_k: int = 5,
                 rerank: bool = False):
        """
        Initialize RAG pipeline.
        
        Args:
            search_engine: SemanticSearch instance
            llm_client: LLM client for generation
            top_k: Number of documents to retrieve
            rerank: Enable reranking
        """
        self.search_engine = search_engine
        self.llm_client = llm_client
        self.top_k = top_k
        self.rerank = rerank
        
        # Reranker
        self._reranker = None
        
        # Custom prompt template
        self._prompt_template = self.SYSTEM_PROMPT
        
        # Pre/post processors
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []
        
        logger.info("RAG Pipeline initialized")
    
    def set_prompt_template(self, template: str):
        """Set custom prompt template."""
        self._prompt_template = template
    
    def add_preprocessor(self, func: Callable[[str], str]):
        """Add query preprocessor."""
        self._preprocessors.append(func)
    
    def add_postprocessor(self, func: Callable[[str], str]):
        """Add response postprocessor."""
        self._postprocessors.append(func)
    
    async def query(self, question: str,
                    filters: Dict = None) -> RAGResponse:
        """
        Query RAG pipeline.
        
        Args:
            question: User question
            filters: Search filters
            
        Returns:
            RAGResponse
        """
        start_time = time.time()
        
        # Preprocess question
        processed_question = question
        for preprocessor in self._preprocessors:
            processed_question = preprocessor(processed_question)
        
        # Retrieve relevant documents
        contexts = await self._retrieve(processed_question, filters)
        
        # Rerank if enabled
        if self.rerank and len(contexts) > 1:
            contexts = await self._rerank_contexts(processed_question, contexts)
        
        # Generate response
        answer, tokens = await self._generate(processed_question, contexts)
        
        # Postprocess response
        for postprocessor in self._postprocessors:
            answer = postprocessor(answer)
        
        latency = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            contexts=contexts,
            model=self._get_model_name(),
            latency_ms=latency,
            tokens_used=tokens
        )
    
    async def _retrieve(self, question: str,
                        filters: Dict = None) -> List[RAGContext]:
        """Retrieve relevant contexts."""
        if self.search_engine is None:
            return []
        
        results = self.search_engine.search(
            question,
            top_k=self.top_k,
            filters=filters
        )
        
        contexts = []
        for result in results:
            contexts.append(RAGContext(
                content=result.content,
                source=result.doc_id,
                score=result.score,
                metadata=result.metadata
            ))
        
        return contexts
    
    async def _rerank_contexts(self, question: str,
                               contexts: List[RAGContext]) -> List[RAGContext]:
        """Rerank retrieved contexts."""
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = [(question, ctx.content) for ctx in contexts]
            scores = self._reranker.predict(pairs)
            
            # Update scores and sort
            for ctx, score in zip(contexts, scores):
                ctx.score = float(score)
            
            contexts.sort(key=lambda x: x.score, reverse=True)
            
        except ImportError:
            pass
        
        return contexts[:self.top_k]
    
    async def _generate(self, question: str,
                        contexts: List[RAGContext]) -> tuple:
        """Generate response using LLM."""
        # Build context string
        context_str = "\n\n".join([
            f"[{i+1}] {ctx.content}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Build prompt
        prompt = self._prompt_template.format(
            context=context_str,
            question=question
        )
        
        # Generate
        if self.llm_client:
            try:
                response = await self.llm_client.generate(prompt)
                return response.text, response.tokens_used
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
        
        # Fallback response
        if contexts:
            answer = f"Based on the available information:\n\n{contexts[0].content}"
        else:
            answer = "I don't have enough information to answer this question."
        
        return answer, 0
    
    def _get_model_name(self) -> str:
        """Get LLM model name."""
        if self.llm_client and hasattr(self.llm_client, 'model'):
            return self.llm_client.model
        return "unknown"
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to search index."""
        if self.search_engine:
            self.search_engine.add_documents(documents)
    
    async def query_with_sources(self, question: str) -> Dict:
        """Query with formatted sources."""
        response = await self.query(question)
        
        sources = []
        for i, ctx in enumerate(response.contexts):
            sources.append({
                "number": i + 1,
                "source": ctx.source,
                "content": ctx.content[:200] + "...",
                "relevance": ctx.score
            })
        
        return {
            "answer": response.answer,
            "sources": sources,
            "model": response.model,
            "latency_ms": response.latency_ms
        }
    
    async def chat(self, messages: List[Dict],
                   use_rag: bool = True) -> RAGResponse:
        """
        Chat with RAG.
        
        Args:
            messages: Chat history
            use_rag: Whether to use RAG
            
        Returns:
            RAGResponse
        """
        # Get last user message
        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break
        
        if use_rag and question:
            return await self.query(question)
        
        # Direct LLM response
        if self.llm_client:
            response = await self.llm_client.chat(messages)
            return RAGResponse(
                answer=response.text,
                contexts=[],
                model=self._get_model_name(),
                latency_ms=0,
                tokens_used=response.tokens_used
            )
        
        return RAGResponse(
            answer="No LLM configured.",
            contexts=[],
            model="none",
            latency_ms=0
        )
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "top_k": self.top_k,
            "rerank_enabled": self.rerank,
            "documents": self.search_engine.count() if self.search_engine else 0,
            "preprocessors": len(self._preprocessors),
            "postprocessors": len(self._postprocessors)
        }
    
    def __repr__(self) -> str:
        return f"RAGPipeline(top_k={self.top_k}, rerank={self.rerank})"
