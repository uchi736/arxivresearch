"""
PDF translation node implementation
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

from src.utils.pdf_translator import PDFTranslatorWithReportLab



def translate_pdfs_node(state: AdvancedAgentState, container: AppContainer):
    """Translate PDFs to Japanese"""
    logger.info("--- PDF翻訳を開始 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    
    translator = PDFTranslatorWithReportLab()
    
    # Limit to 5 papers max
    papers_to_translate = state["found_papers"][:5]
    
    # Start the step
    tracker.start_step("論文翻訳", {
        "total_items": len(papers_to_translate)
    })
    
    for i, paper in enumerate(papers_to_translate):
        logger.info(f"  翻訳中 ({i+1}/{len(papers_to_translate)}): {paper['title']}")
        
        # Update progress
        tracker.update_step("論文翻訳",
                          f"翻訳中: {paper['title'][:50]}...",
                          {"completed_items": i})
        
        try:
            # Download PDF to temp file
            response = requests.get(paper["pdf_url"], timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            # Execute translation
            output_filename = f"{paper['arxiv_id'].replace('.', '_')}_translated.pdf"
            translator.translate_pdf(tmp_file_path, output_filename)
            
            os.unlink(tmp_file_path)
            logger.debug(f"  → 翻訳済みPDFを 'outputs/{output_filename}' に保存しました")

        except Exception as e:
            logger.debug(f"  → 翻訳エラー: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            continue
    
    # Complete the step
    tracker.complete_step("論文翻訳", {
        "translated_papers": len(papers_to_translate)
    })
    
    return {"progress_tracker": save_progress_tracker(tracker)}


