"""
Full text processing node implementation
"""

import json
import os
import requests
import tempfile
from datetime import datetime
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from src.core.models import (
    AdvancedAgentState, PaperMemory, ResearchPlan, SearchQuery,
    OchiaiFormatAdvanced, OCHIAI_SECTION_MAPPING, ImprovedResearchPlan, PaperMetadata
)
from src.core.dependencies import AppContainer
from .base import (
    logger,
    get_progress_tracker, save_progress_tracker, paper_memories
)

from src.analysis.simple_pdf_processor import SimplePDFProcessor
from src.analysis.unified_processor import UnifiedPaperProcessor




def advanced_fulltext_processing_node(state: AdvancedAgentState, container: AppContainer):
    """Advanced full text processing node"""
    logger.info("--- 高度な全文処理を開始 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    
    paper_memories.clear()
    
    # Adjust number of papers based on analysis depth
    analysis_config = container.config.analysis
    depth = state["research_plan"].get("analysis_depth", "moderate")
    # Get max papers based on depth
    if depth == "shallow":
        max_papers = analysis_config.max_papers_shallow
    elif depth == "moderate":
        max_papers = analysis_config.max_papers_moderate
    else:
        max_papers = analysis_config.max_papers_deep
    
    papers_to_process = state["found_papers"][:max_papers]
    
    # Start the step
    tracker.start_step("フルテキスト処理", {
        "total_items": len(papers_to_process)
    })
    
    for i, paper in enumerate(papers_to_process):
        logger.info(f"\n処理中 ({i+1}/{len(papers_to_process)}): {paper['title']}")
        
        # Update progress
        tracker.update_step("フルテキスト処理",
                          f"処理中: {paper['title'][:50]}...",
                          {"completed_items": i})
        
        try:
            # Use unified processor with format preference
            paper_format = state.get("paper_format", "auto")
            processor = UnifiedPaperProcessor(prefer_html=True)
            pdf_data = processor.process_paper(paper["arxiv_id"], format_preference=paper_format)
            if not pdf_data:
                logger.debug("  → PDF処理失敗、スキップ")
                continue
            
            # Save full text directly to paper memory
            # Get full text from unified processor result
            full_text = pdf_data.get("text", pdf_data.get("full_text", ""))
            
            paper_memory = PaperMemory(
                paper_id=paper["arxiv_id"],
                sections={},
                claims=[],
                chunks=[],
                coverage_map={},
                token_budget_used=0,
                full_text=full_text  # Store full text for Map-Reduce analysis
            )
            
            paper_memories[paper["arxiv_id"]] = paper_memory
            
        except Exception as e:
            logger.debug(f"  → 処理エラー: {e}")
            continue
    
    # Complete the step
    tracker.complete_step("フルテキスト処理", {
        "processed_papers": len(paper_memories)
    })
    
    return {"progress_tracker": save_progress_tracker(tracker)}
