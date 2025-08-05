"""
Pydantic models and data structures for arXiv Research Agent

Centralized definition of all data models used throughout the application.
"""

from typing import TypedDict, Dict, Optional, Literal, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np


# --- TypedDict for LangGraph State ---
class AdvancedAgentState(TypedDict):
    initial_query: str
    research_plan: Optional[Dict]
    search_queries: List[Dict]
    found_papers: List[Dict]
    analyzed_papers: List[Dict]
    final_report: str
    token_budget: int
    analysis_mode: str
    total_tokens_used: int
    progress_tracker: Optional[Dict]  # Progress tracking information


# --- Data Classes ---
@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    text: str
    section: str
    page_num: int
    chunk_id: str
    embedding: Optional[np.ndarray] = None


# --- Pydantic Models ---
class PaperMetadata(BaseModel):
    """Paper metadata from arXiv"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    pdf_url: str
    published_date: str
    categories: List[str]


class SectionContent(BaseModel):
    """Section content with metadata"""
    section_name: str
    content: str
    page_range: Tuple[int, int]
    chunk_ids: List[str]


class PaperMemory(BaseModel):
    """Paper memory state for processing"""
    paper_id: str
    sections: Dict[str, SectionContent]
    chunks: List[DocumentChunk]
    claims: List[Any] = Field(default_factory=list)
    coverage_map: Dict[str, float] = Field(default_factory=dict)
    token_budget_used: int = Field(default=0)
    full_text: Optional[str] = Field(default=None)  # Full text for Map-Reduce analysis

    class Config:
        arbitrary_types_allowed = True


class SearchQuery(BaseModel):
    """arXiv search query configuration"""
    keywords: List[str]
    max_results: int = Field(default=5, gt=0)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(default="relevance")
    category: Optional[str] = Field(default=None)


class ResearchPlan(BaseModel):
    """Research investigation plan"""
    main_topic: str
    sub_topics: List[str]
    num_papers: int = Field(gt=0)
    focus_areas: List[str]
    full_text_analysis: bool = Field(default=True)
    analysis_depth: Literal["shallow", "moderate", "deep"] = Field(default="moderate")


class ImprovedResearchPlan(BaseModel):
    """Enhanced research plan with multilingual support and dynamic parameters"""
    # Original query information
    original_query: str = Field(..., description="Original user query (can be in Japanese)")
    translated_query: str = Field(..., description="English translation of the query")
    query_language: str = Field(default="ja", description="Language code of original query")
    
    # Search strategy
    search_keywords: Dict[str, List[str]] = Field(..., description="Categorized search keywords")
    synonyms: Dict[str, List[str]] = Field(default_factory=dict, description="Synonyms for main keywords")
    arxiv_categories: List[str] = Field(default_factory=list, description="Relevant arXiv categories")
    
    # Dynamic analysis settings
    num_papers: int = Field(ge=1, le=50, description="Number of papers to analyze")
    analysis_depth: Literal["shallow", "moderate", "deep"] = Field(..., description="Analysis depth")
    time_range: Literal["recent", "foundational", "all"] = Field(default="all", description="Time range for papers")
    
    # Legacy compatibility
    main_topic: str = Field(..., description="Main research topic")
    sub_topics: List[str] = Field(..., description="Sub-topics to explore")
    focus_areas: List[str] = Field(..., description="Specific areas to focus on")
    full_text_analysis: bool = Field(default=True, description="Whether to analyze full text")
    
    # Metadata
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence in the plan")
    plan_reasoning: str = Field(default="", description="Reasoning behind the plan")


class OchiaiFormatAdvanced(BaseModel):
    """Advanced Ochiai format analysis"""
    what_is_it: str
    comparison_with_prior_work: str
    key_technique: str
    validation_method: str
    experimental_results: str
    discussion_points: str
    next_papers: List[str]
    implementation_details: str
    why_selected: str
    applicability: str
    evidence_map: Dict[str, List[Tuple[int, str]]]  # Evidence mapping


# --- Implementation Assistant Models ---
class Hyperparameters(BaseModel):
    """Structured hyperparameters extraction"""
    learning_rate: float = Field(..., description="Learning rate for training")
    batch_size: int = Field(..., description="Training batch size")
    optimizer: str = Field(..., description="Optimizer type (e.g., Adam, SGD)")
    epochs: int = Field(..., description="Number of training epochs")
    model_specific: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")

    class Config:
        protected_namespaces = ()


class ReproducibilityItem(BaseModel):
    """Reproducibility checklist item"""
    name: str = Field(..., description="Item name (e.g., 'Dataset Access')")
    is_available: bool = Field(..., description="Whether information is available")
    details: str = Field(..., description="Details about availability")


class ReproducibilityChecklist(BaseModel):
    """Complete reproducibility assessment"""
    items: List[ReproducibilityItem]
    overall_score: float = Field(..., description="Score from 0.0 to 1.0", ge=0.0, le=1.0)


# --- Constants and Mappings ---
SECTION_PATTERNS = [
    r"abstract",
    r"introduction", 
    r"related\s*work|background|prior\s*work",
    r"method(?:s)?|approach|algorithm|model|framework",
    r"experiment(?:s)?|evaluation|setup",
    r"result(?:s)?|finding(?:s)?",
    r"discussion|analysis",
    r"conclusion(?:s)?|summary",
    r"limitation(?:s)?|future\s*work",
    r"appendix|supplementary"
]

# Ochiai format mapping to sections
OCHIAI_SECTION_MAPPING = {
    "what_is_it": {
        "sections": ["abstract", "introduction"],
        "weight": 0.20
    },
    "comparison_with_prior_work": {
        "sections": ["related_work", "introduction"],
        "weight": 0.10
    },
    "key_technique": {
        "sections": ["method", "approach", "algorithm"],
        "weight": 0.35
    },
    "validation_method": {
        "sections": ["experiment", "setup", "evaluation"],
        "weight": 0.15
    },
    "experimental_results": {
        "sections": ["results", "findings", "ablation"],
        "weight": 0.25
    },
    "discussion_points": {
        "sections": ["discussion", "limitation", "future_work"],
        "weight": 0.10
    },
    "implementation_details": {
        "sections": ["method", "appendix", "algorithm"],
        "weight": 0.20
    }
}
