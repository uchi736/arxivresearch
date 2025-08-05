"""
Structured analysis node with Map-Reduce logic
"""
import time
import re
from typing import Dict, List
from src.core.models import AdvancedAgentState, OchiaiFormatAdvanced, OCHIAI_SECTION_MAPPING
from src.core.dependencies import AppContainer
from .base import logger, get_progress_tracker, save_progress_tracker

def structured_analysis_node(state: AdvancedAgentState, container: AppContainer):
    """
    Analyzes papers using a Map-Reduce approach to generate structured data.
    """
    logger.info("--- 構造化分析を開始 (Map-Reduce) ---")
    start_time = time.time()
    
    tracker = get_progress_tracker(state)
    tracker.start_step("論文分析", {"total_items": len(state['found_papers'])})
    
    llm_model = container.llm_model
    analyzed_papers = []

    for idx, paper in enumerate(state['found_papers']):
        full_text = paper.get("full_text", "")
        if not full_text:
            logger.warning(f"Skipping paper {paper['arxiv_id']} due to missing full text.")
            continue

        # --- Map Step ---
        section_summaries = {}
        # A simple way to split by sections. A more robust implementation would use regex.
        raw_sections = full_text.split('\n\n') 
        
        # This is a simplified section mapping.
        # A more robust implementation would use the patterns from models.py
        # and handle the text more intelligently.
        # For now, we just create a prompt for each section of the Ochiai format.
        
        map_prompts = []
        for key, mapping in OCHIAI_SECTION_MAPPING.items():
            # A simple prompt for the section
            prompt = f"""
            Paper Title: {paper['title']}
            
            Based on the full text of the paper, provide a summary for the following section:
            - {key} ({', '.join(mapping['sections'])})
            
            Full Text (first 10000 chars):
            {full_text[:10000]}
            
            Summary:
            """
            map_prompts.append(prompt)

        try:
            # In a real implementation, we would run these prompts in parallel.
            # For simplicity, we run them sequentially here.
            map_results = [llm_model.invoke(p) for p in map_prompts]
            
            # --- Reduce Step ---
            combined_summary = ""
            for i, (key, _) in enumerate(OCHIAI_SECTION_MAPPING.items()):
                combined_summary += f"## {key}\n{map_results[i].content}\n\n"

            reduce_prompt = f"""
            Based on the following section summaries, generate a final analysis in the Ochiai format.
            
            Summaries:
            {combined_summary}
            
            Please generate a JSON object that conforms to the OchiaiFormatAdvanced model.
            The JSON object should have the following keys: {', '.join(OchiaiFormatAdvanced.model_fields.keys())}
            """
            
            reduce_response = llm_model.invoke(reduce_prompt)
            
            # For now, we'll just put the raw text in the analysis.
            # A more robust solution would parse the JSON and validate it.
            analysis_content = reduce_response.content
            tokens_used = sum(r.response_metadata.get("usage", {}).get("total_tokens", 0) for r in map_results)
            tokens_used += reduce_response.response_metadata.get("usage", {}).get("total_tokens", 0)

            analyzed_paper = {
                "metadata": paper,
                "analysis": {"summary": analysis_content}, # Placeholder
                "analysis_type": "structured_map_reduce",
                "tokens_used": tokens_used
            }
            analyzed_papers.append(analyzed_paper)
            tracker.update_step("論文分析", f"解析完了: {paper['arxiv_id']}", {"completed_items": idx + 1})

        except Exception as e:
            logger.error(f"Error analyzing paper {paper['arxiv_id']}: {e}")
            tracker.update_step("論文分析", f"エラー: {paper['arxiv_id']}", {"completed_items": idx + 1, "errors": 1})

    end_time = time.time()
    logger.info(f"分析フェーズ全体が {end_time - start_time:.2f}秒で完了しました。")
    
    tracker.complete_step("論文分析", {"analyzed_papers": len(analyzed_papers)})
    
    return {
        "analyzed_papers": analyzed_papers,
        "progress_tracker": save_progress_tracker(tracker)
    }
