"""
Results saving node implementation
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


def save_results_node(state: AdvancedAgentState, container: AppContainer):
    """Save results node"""
    logger.info("--- 結果を保存中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("結果保存")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/arxiv_research_advanced_{timestamp}.json"
    
    os.makedirs("reports", exist_ok=True)
    
    total_tokens = sum(p.get('tokens_used', 0) for p in state["analyzed_papers"])
    
    # Prepare save data
    save_data = {
        "query": state["initial_query"],
        "research_plan": state["research_plan"],
        "analysis_mode": state["analysis_mode"],
        "analyzed_papers": state["analyzed_papers"],
        "report": state["final_report"],
        "timestamp": timestamp,
        "total_tokens_used": total_tokens
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    # Also save the report as markdown
    report_filename = f"reports/arxiv_advanced_report_{timestamp}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(state["final_report"])
    
    # Also save the report as HTML
    from src.utils.html_report_generator import HTMLReportGenerator
    html_generator = HTMLReportGenerator()
    html_filename = f"reports/arxiv_advanced_report_{timestamp}.html"
    html_generator.save_html_report(
        state["final_report"], 
        state["initial_query"], 
        html_filename,
        datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    )
    
    logger.info(f"--- 結果を {filename} に保存しました ---")
    logger.info(f"--- レポートを {report_filename} に保存しました ---")
    logger.info(f"--- HTMLレポートを {html_filename} に保存しました ---")
    
    # Complete the step
    tracker.complete_step("結果保存", {
        "json_file": filename,
        "report_file": report_filename,
        "html_file": html_filename
    })
    
    # Mark workflow as complete
    tracker.is_complete = True
    
    return {
        "total_tokens_used": total_tokens,
        "progress_tracker": save_progress_tracker(tracker)
    }
