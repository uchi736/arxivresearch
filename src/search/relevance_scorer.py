"""
Relevance scoring system for arXiv paper search results

This module provides sophisticated scoring mechanisms to rank papers
based on their relevance to the search query and research plan.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher

from src.core.models import PaperMetadata, ImprovedResearchPlan


class RelevanceScorer:
    """Calculate relevance scores for papers based on improved research plan"""
    
    # Score weights definition
    WEIGHTS = {
        # Keyword matching
        "title_exact_match": 3.0,      # Exact match in title
        "title_partial_match": 2.0,    # Partial match in title
        "abstract_match": 1.0,         # Match in abstract
        "abstract_density": 0.5,       # Keyword density in abstract
        
        # Category/field matching
        "primary_category_match": 2.0,  # Primary category match
        "sub_category_match": 1.0,      # Sub-category match
        
        # Temporal elements
        "recency_bonus": 1.5,          # Recent papers bonus
        "foundational_bonus": 2.0,     # Foundational papers bonus
        
        # Query type specific
        "core_keyword_match": 2.5,     # Core keywords
        "technical_keyword_match": 2.0, # Technical terms
        "method_keyword_match": 1.5,    # Method names
        "application_keyword_match": 1.0, # Application areas
        
        # Penalties
        "title_length_penalty": -0.5,   # Title too long
        "abstract_short_penalty": -1.0, # Abstract too short
        "noise_word_penalty": -3.0,     # Contains noise words
    }
    
    # Category importance weights
    CATEGORY_WEIGHTS = {
        "core": 1.2,       # Core keywords are most important
        "technical": 1.0,   # Technical terms are standard
        "method": 0.9,      # Methods slightly lower
        "application": 0.8  # Applications are supplementary
    }
    
    # Noise words that indicate low quality
    NOISE_WORDS = [
        "withdrawn", "error", "correction", "retracted", 
        "erratum", "comment on", "reply to", "response to"
    ]
    
    def __init__(self):
        """Initialize the relevance scorer"""
        pass
    
    def calculate_score(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> Tuple[float, Dict]:
        """
        Calculate total relevance score for a paper
        
        Returns:
            Tuple of (total_score, score_details)
        """
        score = 0.0
        details = {}
        
        # 1. Keyword matching score
        keyword_score = self._calculate_keyword_score(paper, improved_plan)
        score += keyword_score
        details['keyword'] = keyword_score
        
        # 2. Category matching score
        category_score = self._calculate_category_score(paper, improved_plan)
        score += category_score
        details['category'] = category_score
        
        # 3. Temporal relevance score
        temporal_score = self._calculate_temporal_score(paper, improved_plan)
        score += temporal_score
        details['temporal'] = temporal_score
        
        # 4. Quality penalties
        penalty = self._calculate_penalties(paper)
        score += penalty
        details['penalty'] = penalty
        
        # 5. Synonym matching bonus
        synonym_score = self._calculate_synonym_score(paper, improved_plan)
        score += synonym_score
        details['synonym'] = synonym_score
        
        # Ensure non-negative score
        score = max(0.0, score)
        
        return score, details
    
    def _calculate_keyword_score(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> float:
        """Calculate keyword-based score"""
        score = 0.0
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()
        
        # Check keywords by category
        for category, keywords in improved_plan.search_keywords.items():
            category_weight = self.CATEGORY_WEIGHTS.get(category, 1.0)
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Title matching
                if keyword_lower in title_lower:
                    if self._is_exact_word_match(keyword_lower, title_lower):
                        score += self.WEIGHTS["title_exact_match"] * category_weight
                    else:
                        score += self.WEIGHTS["title_partial_match"] * category_weight
                
                # Abstract matching
                if keyword_lower in abstract_lower:
                    # Consider keyword occurrence count
                    count = abstract_lower.count(keyword_lower)
                    density_bonus = min(count * 0.2, 1.0)  # Max 1.0
                    score += self.WEIGHTS["abstract_match"] * category_weight
                    score += self.WEIGHTS["abstract_density"] * density_bonus * category_weight
        
        return score
    
    def _calculate_synonym_score(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> float:
        """Calculate synonym matching score"""
        score = 0.0
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()
        
        for main_term, synonyms in improved_plan.synonyms.items():
            main_term_lower = main_term.lower()
            
            # Main term in title gets bonus
            if main_term_lower in title_lower:
                score += 1.0
            
            # Synonyms get partial credit
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                if synonym_lower in title_lower:
                    score += 0.5
                elif synonym_lower in abstract_lower:
                    score += 0.2
        
        return score
    
    def _calculate_temporal_score(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> float:
        """Calculate temporal relevance score"""
        score = 0.0
        
        # Parse publication date
        try:
            publish_date = datetime.fromisoformat(paper.published_date.replace('Z', '+00:00'))
            days_ago = (datetime.now() - publish_date).days
        except:
            # If date parsing fails, assume neutral
            return score
        
        if improved_plan.time_range == "recent":
            # Favor recent papers (exponential decay)
            if days_ago < 180:  # Within 6 months
                score += self.WEIGHTS["recency_bonus"] * 1.0
            elif days_ago < 365:  # Within 1 year
                score += self.WEIGHTS["recency_bonus"] * 0.7
            elif days_ago < 730:  # Within 2 years
                score += self.WEIGHTS["recency_bonus"] * 0.3
                
        elif improved_plan.time_range == "foundational":
            # Favor classic papers (3+ years old)
            if days_ago > 1095:  # More than 3 years
                age_factor = min(days_ago / 3650, 1.0)  # Max at 10 years
                score += self.WEIGHTS["foundational_bonus"] * age_factor
        
        return score
    
    def _calculate_category_score(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> float:
        """Calculate arXiv category matching score with domain filtering"""
        score = 0.0
        
        if not improved_plan.arxiv_categories:
            # If no categories specified, apply domain-specific filtering
            return self._apply_domain_filtering(paper, improved_plan)
        
        paper_categories = set(paper.categories)
        target_categories = set(improved_plan.arxiv_categories)
        
        # First category is primary category
        if paper.categories and paper.categories[0] in target_categories:
            score += self.WEIGHTS["primary_category_match"]
        
        # Other category matches
        common_categories = paper_categories & target_categories
        if len(common_categories) > 1:  # Don't double count primary
            score += (len(common_categories) - 1) * self.WEIGHTS["sub_category_match"]
        
        # Apply domain filtering penalties
        domain_penalty = self._apply_domain_filtering(paper, improved_plan)
        score += domain_penalty
        
        return score
    
    def _apply_domain_filtering(self, paper: PaperMetadata, improved_plan: ImprovedResearchPlan) -> float:
        """Apply domain-specific filtering penalties"""
        penalty = 0.0
        
        # Check if this is an AI agent evaluation query
        is_agent_eval = any(term in improved_plan.translated_query.lower() 
                           for term in ["agent", "evaluation", "assessment", "benchmark"])
        
        if is_agent_eval:
            # Define irrelevant categories for AI agent evaluation
            irrelevant_categories = {
                "cond-mat", "physics", "astro-ph", "nucl", "hep", "gr-qc",
                "quant-ph", "math-ph", "nlin", "chao-dyn", "comp-gas",
                "q-bio", "q-fin", "econ", "stat.AP"  # Applied stats often not AI
            }
            
            # Check primary category
            if paper.categories and any(cat.startswith(irrelevant) 
                                     for irrelevant in irrelevant_categories 
                                     for cat in paper.categories):
                penalty -= 5.0  # Heavy penalty for wrong domain
            
            # Check for battery/materials science keywords in title
            battery_keywords = ["battery", "intercalation", "electrode", "electrolyte", 
                              "lithium", "sodium", "materials", "crystal", "alloy"]
            if any(keyword in paper.title.lower() for keyword in battery_keywords):
                penalty -= 8.0  # Very heavy penalty for battery papers
            
            # Check for other non-AI domains in title
            non_ai_domains = ["publication metric", "bibliometric", "h-index", 
                            "citation analysis", "impact factor", "journal ranking"]
            if any(domain in paper.title.lower() for domain in non_ai_domains):
                penalty -= 6.0  # Heavy penalty for publication metrics
        
        return penalty
    
    def _calculate_penalties(self, paper: PaperMetadata) -> float:
        """Calculate quality-based penalties"""
        penalty = 0.0
        
        # Title too long (word count)
        title_words = len(paper.title.split())
        if title_words > 20:
            penalty += self.WEIGHTS["title_length_penalty"]
        
        # Abstract too short
        if len(paper.abstract) < 200:
            penalty += self.WEIGHTS["abstract_short_penalty"]
        
        # Noise word detection
        title_lower = paper.title.lower()
        for noise in self.NOISE_WORDS:
            if noise in title_lower:
                penalty += self.WEIGHTS["noise_word_penalty"]
                break  # Only apply once
        
        return penalty
    
    def _is_exact_word_match(self, keyword: str, text: str) -> bool:
        """Check if keyword appears as exact word in text"""
        # Use word boundaries for exact matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))


def remove_duplicates_smart(papers: List[PaperMetadata], similarity_threshold: float = 0.85) -> List[PaperMetadata]:
    """
    Remove duplicates with smart title similarity checking
    
    Args:
        papers: List of papers to deduplicate
        similarity_threshold: Threshold for title similarity (0-1)
    
    Returns:
        List of unique papers
    """
    unique_papers = []
    seen_ids = set()
    seen_titles = []
    
    for paper in papers:
        # Check ID duplication
        if paper.arxiv_id in seen_ids:
            continue
        
        # Check title similarity
        title_normalized = normalize_title(paper.title)
        is_duplicate = False
        
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title_normalized, seen_title).ratio()
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_ids.add(paper.arxiv_id)
            seen_titles.append(title_normalized)
            unique_papers.append(paper)
    
    return unique_papers


def normalize_title(title: str) -> str:
    """Normalize title for comparison"""
    # Convert to lowercase
    title = title.lower()
    
    # Remove common suffixes
    suffixes = [
        " - a survey",
        " - a review", 
        " - an overview",
        " (extended version)",
        " (preprint)",
        " (draft)"
    ]
    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)]
    
    # Remove extra whitespace
    title = ' '.join(title.split())
    
    return title


def filter_by_time_range(papers: List[PaperMetadata], time_range: str) -> List[PaperMetadata]:
    """
    Filter papers based on time range preference
    
    Args:
        papers: List of papers to filter
        time_range: One of "recent", "foundational", "all"
    
    Returns:
        Filtered list of papers
    """
    if time_range == "all":
        return papers
    
    filtered = []
    now = datetime.now()
    
    for paper in papers:
        try:
            publish_date = datetime.fromisoformat(paper.published_date.replace('Z', '+00:00'))
            days_ago = (now - publish_date).days
            
            if time_range == "recent":
                # Keep papers from last 2 years
                if days_ago <= 730:
                    filtered.append(paper)
            
            elif time_range == "foundational":
                # Prefer papers older than 3 years
                if days_ago > 1095:
                    filtered.append(paper)
                # But include some recent highly relevant papers
                elif hasattr(paper, 'relevance_score') and paper.relevance_score > 5.0:
                    filtered.append(paper)
            
        except:
            # If date parsing fails, include the paper
            filtered.append(paper)
    
    return filtered


def filter_quality(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    """
    Filter out low quality papers
    
    Args:
        papers: List of papers to filter
    
    Returns:
        List of quality papers
    """
    filtered = []
    
    for paper in papers:
        # Abstract too short
        if len(paper.abstract) < 100:
            continue
        
        # Title contains spam indicators
        title_lower = paper.title.lower()
        spam_indicators = ["test", "untitled", "no title", "placeholder"]
        if any(spam in title_lower for spam in spam_indicators):
            continue
        
        # Paper is withdrawn or has errors
        if any(word in title_lower for word in ["withdrawn", "retracted"]):
            continue
        
        filtered.append(paper)
    
    return filtered