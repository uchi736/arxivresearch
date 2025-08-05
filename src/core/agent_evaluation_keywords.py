"""
Specialized keyword generation for AI agent evaluation queries

This module provides domain-specific keyword generation to improve
search results for agent evaluation related queries.
"""

from typing import Dict, List, Tuple
import re


class AgentEvaluationKeywordGenerator:
    """Generate optimized keywords for agent evaluation searches"""
    
    # Core evaluation terms
    EVALUATION_CORE_TERMS = [
        "agent evaluation",
        "AI agent",
        "multi-agent",
        "autonomous agent",
        "intelligent agent"
    ]
    
    # Evaluation methodology terms
    EVALUATION_METHODS = [
        "benchmark",
        "evaluation metric",
        "performance metric", 
        "evaluation framework",
        "assessment method",
        "evaluation methodology"
    ]
    
    # Specific agent types
    AGENT_TYPES = [
        "LLM agent",
        "conversational agent",
        "dialogue agent",
        "chatbot",
        "virtual assistant",
        "task-oriented agent",
        "reinforcement learning agent",
        "cognitive agent"
    ]
    
    # Evaluation aspects
    EVALUATION_ASPECTS = [
        "human evaluation",
        "automated evaluation",
        "user study",
        "A/B testing",
        "comparative evaluation",
        "ablation study",
        "error analysis"
    ]
    
    # Domain-specific applications
    APPLICATIONS = [
        "task completion",
        "dialogue system",
        "code generation",
        "problem solving",
        "decision making",
        "planning",
        "reasoning"
    ]
    
    def __init__(self):
        """Initialize the keyword generator"""
        pass
    
    def is_agent_evaluation_query(self, query: str) -> bool:
        """Check if query is related to agent evaluation"""
        query_lower = query.lower()
        
        # Check for evaluation terms
        evaluation_terms = ["evaluat", "assess", "benchmark", "metric", "measure", "test"]
        has_evaluation = any(term in query_lower for term in evaluation_terms)
        
        # Check for agent terms
        agent_terms = ["agent", "エージェント", "chatbot", "assistant", "ai system"]
        has_agent = any(term in query_lower for term in agent_terms)
        
        return has_evaluation and has_agent
    
    def generate_keywords(self, query: str, translated_query: str = None) -> Dict[str, List[str]]:
        """
        Generate comprehensive keywords for agent evaluation queries
        
        Args:
            query: Original query
            translated_query: English translation if original is non-English
            
        Returns:
            Dictionary of categorized keywords
        """
        # Use translated query if available
        working_query = translated_query or query
        
        keywords = {
            "core": [],
            "technical": [],
            "application": [],
            "method": []
        }
        
        # Core keywords - always include these for agent evaluation
        keywords["core"] = [
            "agent evaluation",
            "AI agent",
            "evaluation metric"
        ]
        
        # Add specific terms based on query content
        query_lower = working_query.lower()
        
        # Check for LLM/language model agents
        if any(term in query_lower for term in ["llm", "language model", "gpt", "chatgpt"]):
            keywords["core"].append("LLM agent")
            keywords["technical"].extend([
                "LLM evaluation",
                "language model agent",
                "LLM benchmark"
            ])
        
        # Check for multi-agent
        if "multi" in query_lower or "team" in query_lower:
            keywords["core"].append("multi-agent")
            keywords["technical"].append("multi-agent evaluation")
        
        # Check for specific evaluation types
        if "benchmark" in query_lower:
            keywords["technical"].extend([
                "agent benchmark",
                "evaluation benchmark",
                "standardized evaluation"
            ])
        
        # Add method keywords
        keywords["method"] = [
            "evaluation framework",
            "performance metric",
            "assessment method"
        ]
        
        # Add application keywords based on context
        if "conversation" in query_lower or "dialogue" in query_lower:
            keywords["application"].extend([
                "conversational agent",
                "dialogue evaluation"
            ])
        elif "task" in query_lower:
            keywords["application"].extend([
                "task-oriented agent",
                "task completion"
            ])
        else:
            # General applications
            keywords["application"] = [
                "autonomous agent",
                "intelligent system"
            ]
        
        # Remove duplicates while preserving order
        for category in keywords:
            keywords[category] = list(dict.fromkeys(keywords[category]))
        
        return keywords
    
    def create_search_queries(self, keywords: Dict[str, List[str]], num_papers: int = 10) -> List[Tuple[List[str], int]]:
        """
        Create optimized search queries from keywords
        
        Args:
            keywords: Categorized keywords
            num_papers: Total number of papers to find
            
        Returns:
            List of (keyword_list, max_results) tuples
        """
        queries = []
        
        # Strategy 1: Core evaluation terms (broad search)
        if keywords["core"]:
            queries.append(
                ([keywords["core"][0]], min(5, num_papers))
            )
        
        # Strategy 2: Technical terms (specific methods)
        if len(keywords["technical"]) >= 2:
            queries.append(
                (keywords["technical"][:2], min(3, num_papers // 2))
            )
        
        # Strategy 3: Combined core + method
        if keywords["core"] and keywords["method"]:
            queries.append(
                ([keywords["core"][1], keywords["method"][0]], min(3, num_papers // 3))
            )
        
        # Strategy 4: Application specific
        if keywords["application"]:
            queries.append(
                (keywords["application"][:2], min(2, num_papers // 4))
            )
        
        # Ensure we don't exceed total papers
        total_requested = sum(q[1] for q in queries)
        if total_requested > num_papers and queries:
            # Adjust the first query
            queries[0] = (queries[0][0], queries[0][1] - (total_requested - num_papers))
        
        return queries
    
    def enhance_existing_keywords(self, existing_keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Enhance existing keywords with agent evaluation specific terms
        
        Args:
            existing_keywords: Current keyword dictionary
            
        Returns:
            Enhanced keyword dictionary
        """
        enhanced = existing_keywords.copy()
        
        # Add essential evaluation terms if missing
        if "evaluation" not in ' '.join(str(kw) for kws in enhanced.values() for kw in kws).lower():
            enhanced.setdefault("core", []).append("evaluation")
        
        # Add benchmark if discussing metrics
        if any("metric" in str(kw).lower() for kws in enhanced.values() for kw in kws):
            enhanced.setdefault("technical", []).append("benchmark")
        
        # Ensure we have agent-related terms
        has_agent = any("agent" in str(kw).lower() for kws in enhanced.values() for kw in kws)
        if not has_agent:
            enhanced.setdefault("core", []).insert(0, "AI agent")
        
        return enhanced