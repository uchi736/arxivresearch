"""
Advanced arXiv research workflow builder
"""
from functools import partial
from langgraph.graph import StateGraph, END
from src.core.models import AdvancedAgentState
from src.core.dependencies import get_container
from src.workflow.nodes import (
    plan_research_advanced_node,
    generate_queries_node,
    search_papers_node,
    advanced_fulltext_processing_node,
    generate_advanced_report_node,
    save_results_node
)
from src.workflow.nodes.gemini_analysis_node import gemini_full_text_analysis_node
from src.core.config import get_model_config


def build_advanced_workflow():
    """Build the advanced analysis workflow"""
    workflow = StateGraph(AdvancedAgentState)
    
    # Get dependency container
    container = get_container()
    
    # Create partial functions with the container
    plan_research_with_deps = partial(plan_research_advanced_node, container=container)
    generate_queries_with_deps = partial(generate_queries_node, container=container)
    search_papers_with_deps = partial(search_papers_node, container=container)
    process_fulltext_with_deps = partial(advanced_fulltext_processing_node, container=container)
    analyze_with_deps = partial(gemini_full_text_analysis_node, container=container)
    generate_report_with_deps = partial(generate_advanced_report_node, container=container)
    save_results_with_deps = partial(save_results_node, container=container)
    
    # Add nodes (lightweight version - no translation)
    workflow.add_node("plan_research", plan_research_with_deps)
    workflow.add_node("generate_queries", generate_queries_with_deps)
    workflow.add_node("search_papers", search_papers_with_deps)
    workflow.add_node("process_fulltext", process_fulltext_with_deps)
    workflow.add_node("analyze", analyze_with_deps)
    workflow.add_node("generate_report", generate_report_with_deps)
    workflow.add_node("save_results", save_results_with_deps)
    
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
