"""
Advanced arXiv research workflow builder
"""
from functools import partial
from langgraph.graph import StateGraph, END
from src.core.models import AdvancedAgentState
from src.workflow.nodes import (
    plan_research_advanced_node,
    generate_queries_node,
    search_papers_node,
    advanced_fulltext_processing_node,
    exec_summary_node,
    generate_advanced_report_node,
    save_results_node
)
from src.workflow.nodes.gemini_analysis_node import gemini_full_text_analysis_node
from src.core.config import get_model_config


def build_advanced_workflow():
    """Build the advanced analysis workflow"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Clear global state at workflow build time
    from src.workflow.nodes.base import clear_paper_memories
    from src.core.dependencies import clear_container_cache
    
    logger.info("Building new workflow - clearing global state")
    clear_paper_memories()
    clear_container_cache()
    
    workflow = StateGraph(AdvancedAgentState)
    
    # Wrapper functions that get container lazily
    def plan_research_wrapper(state):
        from src.core.dependencies import get_container
        return plan_research_advanced_node(state, get_container())
    
    def generate_queries_wrapper(state):
        from src.core.dependencies import get_container
        return generate_queries_node(state, get_container())
    
    def search_papers_wrapper(state):
        from src.core.dependencies import get_container
        return search_papers_node(state, get_container())
    
    def process_fulltext_wrapper(state):
        from src.core.dependencies import get_container
        return advanced_fulltext_processing_node(state, get_container())
    
    def analyze_wrapper(state):
        from src.core.dependencies import get_container
        return gemini_full_text_analysis_node(state, get_container())
    
    def exec_summary_wrapper(state):
        from src.core.dependencies import get_container
        return exec_summary_node(state, get_container())
    
    def generate_report_wrapper(state):
        from src.core.dependencies import get_container
        return generate_advanced_report_node(state, get_container())
    
    def save_results_wrapper(state):
        from src.core.dependencies import get_container
        return save_results_node(state, get_container())
    
    # Add nodes (lightweight version - no translation)
    workflow.add_node("plan_research", plan_research_wrapper)
    workflow.add_node("generate_queries", generate_queries_wrapper)
    workflow.add_node("search_papers", search_papers_wrapper)
    workflow.add_node("process_fulltext", process_fulltext_wrapper)
    workflow.add_node("analyze", analyze_wrapper)
    workflow.add_node("generate_exec_summary", exec_summary_wrapper)
    workflow.add_node("generate_report", generate_report_wrapper)
    workflow.add_node("save_results", save_results_wrapper)
    
    # Define edges (skip translation step)
    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "generate_queries")
    workflow.add_edge("generate_queries", "search_papers")
    workflow.add_edge("search_papers", "process_fulltext")  # Skip translation
    workflow.add_edge("process_fulltext", "analyze")
    workflow.add_edge("analyze", "generate_exec_summary")
    workflow.add_edge("generate_exec_summary", "generate_report")
    workflow.add_edge("generate_report", "save_results")
    workflow.add_edge("save_results", END)
    
    # Compile without memory saver due to version issues
    return workflow.compile()
