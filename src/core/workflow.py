"""
Advanced arXiv research workflow builder
"""
from langgraph.graph import StateGraph, END
from src.core.models import AdvancedAgentState
from src.analysis.analysis_nodes import (
    plan_research_advanced_node,
    generate_queries_node,
    search_papers_node,
    advanced_fulltext_processing_node,
    ochiai_focused_analysis_node,
    generate_advanced_report_node,
    save_results_node
)
from src.core.config import get_model_config


def build_advanced_workflow():
    """Build the advanced analysis workflow"""
    workflow = StateGraph(AdvancedAgentState)
    
    # Get configuration
    model_config = get_model_config()
    
    # Add nodes (lightweight version - no translation)
    workflow.add_node("plan_research", plan_research_advanced_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("process_fulltext", advanced_fulltext_processing_node)
    workflow.add_node("analyze", ochiai_focused_analysis_node)
    workflow.add_node("generate_report", generate_advanced_report_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define edges (skip translation step)
    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "generate_queries")
    workflow.add_edge("generate_queries", "search_papers")
    workflow.add_edge("search_papers", "process_fulltext")  # Skip translation
    workflow.add_edge("process_fulltext", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", "save_results")
    workflow.add_edge("save_results", END)
    
    # Compile without memory saver due to version issues
    return workflow.compile()