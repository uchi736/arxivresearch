"""
RAG (Retrieval-Augmented Generation) system for paper analysis

This module handles document chunking, embedding computation, and retrieval.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback to using TF-IDF only
    HuggingFaceEmbeddings = None
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.models import DocumentChunk, ExtractedClaim, PaperMemory, SectionContent, SECTION_PATTERNS
from src.core.config import get_rag_config


class PaperRAGSystem:
    """Paper RAG system for advanced document retrieval and analysis"""
    
    def __init__(self, model_name: Optional[str] = None):
        config = get_rag_config()
        
        # Try to initialize HuggingFace embeddings, fall back to TF-IDF only if not available
        if HuggingFaceEmbeddings is not None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name or config.embedding_model
                )
            except Exception as e:
                print(f"Warning: Could not initialize HuggingFace embeddings: {e}")
                print("Falling back to TF-IDF only mode")
                self.embeddings = None
        else:
            print("Warning: HuggingFaceEmbeddings not available, using TF-IDF only")
            self.embeddings = None
            
        self.tfidf = TfidfVectorizer(
            max_features=config.max_features, 
            stop_words='english'
        )
        self.memory: Dict[str, PaperMemory] = {}
        self.config = config
        
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text based on common patterns"""
        sections = {"full_text": text}
        current_section = "abstract"
        current_text = []
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Detect section headers
            matched = False
            for pattern in SECTION_PATTERNS:
                if re.match(r"^\d*\.?\s*" + pattern + r"\s*$", line_lower):
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text)
                    
                    # Start new section
                    current_section = pattern.split(r'\|')[0].replace(r'\s*', '_')
                    current_text = []
                    matched = True
                    break
            
            if not matched:
                current_text.append(line)
        
        # Save last section
        if current_text:
            sections[current_section] = '\n'.join(current_text)
        
        return sections
    
    def chunk_by_section(
        self, 
        sections: Dict[str, str], 
        chunk_size: Optional[int] = None, 
        overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Chunk text by sections"""
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or self.config.chunk_size,
            chunk_overlap=overlap or self.config.chunk_overlap,
            length_function=len,
        )
        
        for section_name, section_text in sections.items():
            if not section_text:
                continue
                
            section_chunks = splitter.split_text(section_text)
            for i, chunk_text in enumerate(section_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    section=section_name,
                    page_num=0,  # Will be updated later
                    chunk_id=f"{section_name}_{i}"
                )
                chunks.append(chunk)
        
        return chunks
    
    def compute_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Compute embeddings for document chunks"""
        if self.embeddings is None:
            # If no embeddings available, use TF-IDF vectors as embeddings
            texts = [chunk.text for chunk in chunks]
            if texts:
                tfidf_matrix = self.tfidf.fit_transform(texts)
                for i, chunk in enumerate(chunks):
                    chunk.embedding = tfidf_matrix[i].toarray().flatten()
        else:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = np.array(embedding)
    
    def mmr_select(
        self, 
        query: str, 
        chunks: List[DocumentChunk], 
        k: int = 5, 
        lambda_param: Optional[float] = None
    ) -> List[DocumentChunk]:
        """Select chunks using Maximal Marginal Relevance"""
        if not chunks:
            return []
        
        if lambda_param is None:
            lambda_param = self.config.mmr_lambda
        
        # Query embedding
        if self.embeddings is None:
            # Use TF-IDF for query embedding
            query_tfidf = self.tfidf.transform([query])
            query_embedding = query_tfidf.toarray().flatten()
        else:
            query_embedding = np.array(self.embeddings.embed_query(query))
        
        # Compute similarities
        chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # MMR selection
        selected_indices = []
        selected_chunks = []
        
        for _ in range(min(k, len(chunks))):
            mmr_scores = []
            
            for i, chunk in enumerate(chunks):
                if i in selected_indices:
                    continue
                
                # Relevance to query
                relevance = similarities[i]
                
                # Maximum similarity to already selected chunks
                if selected_indices:
                    selected_embeddings = chunk_embeddings[selected_indices]
                    max_sim = np.max(cosine_similarity([chunk.embedding], selected_embeddings))
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((i, mmr_score))
            
            if not mmr_scores:
                break
            
            # Select chunk with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            selected_chunks.append(chunks[best_idx])
        
        return selected_chunks
    
    def extract_claims(self, chunk: DocumentChunk, model) -> List[ExtractedClaim]:
        """Extract claims from a chunk using LLM"""
        prompt = f"""
        以下のテキストから重要な主張・発見・手法を箇条書きで抽出してください。
        各項目は簡潔に、数値や具体的な内容を含めてください。
        
        テキスト（{chunk.section}セクション）:
        {chunk.text}
        
        出力形式:
        - 主張1: 具体的な内容
        - 主張2: 具体的な内容
        ...
        """
        
        response = model.invoke(prompt)
        claims = []
        
        # Parse response to extract claims
        for line in response.content.split('\n'):
            if line.strip().startswith('-'):
                claim_text = line.strip()[1:].strip()
                if claim_text:
                    claims.append(ExtractedClaim(
                        claim=claim_text,
                        evidence=chunk.text[:200],  # First 200 chars as evidence
                        section=chunk.section,
                        page_num=chunk.page_num
                    ))
        
        return claims