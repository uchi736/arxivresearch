"""
Gemini-based full-text analysis node
"""
import time
from typing import Dict
from src.core.models import AdvancedAgentState
from src.core.dependencies import AppContainer
from src.analysis.gemini_map_reduce_analyzer import GeminiMapReduceAnalyzer
from .base import logger, get_progress_tracker, save_progress_tracker, paper_memories

def gemini_full_text_analysis_node(state: AdvancedAgentState, container: AppContainer):
    """
    Analyzes the full text of each paper using Gemini Map-Reduce.
    """
    logger.info("--- 全文解析を開始 (Gemini Map-Reduce) ---")
    start_time = time.time()
    
    tracker = get_progress_tracker(state)
    tracker.start_step("論文分析", {"total_items": len(state['found_papers'])})
    
    # Initialize Map-Reduce analyzer
    analyzer = GeminiMapReduceAnalyzer(model=container.llm_model)
    analyzed_papers = []

    for idx, paper in enumerate(state['found_papers']):
        # Get full text from paper memories
        paper_memory = paper_memories.get(paper['arxiv_id'])
        if not paper_memory or not paper_memory.full_text:
            logger.warning(f"Skipping paper {paper['arxiv_id']} due to missing full text.")
            continue

        try:
            # Use Map-Reduce analysis
            logger.info(f"Analyzing paper {idx+1}/{len(state['found_papers'])}: {paper['title'][:60]}...")
            
            # Extract sections from full text
            sections = analyzer.extract_sections(paper_memory.full_text)
            logger.debug(f"  → Extracted {len(sections)} sections")
            
            # Run section analysis (Map phase)
            section_analyses = {}
            for section_name, section_text in sections.items():
                if section_name and section_text:
                    analysis = analyzer.analyze_section(section_name, section_text)
                    if analysis.get("status") != "skipped":
                        section_analyses[section_name] = analysis
            logger.debug(f"  → Completed Map phase with {len(section_analyses)} analyses")
            
            # Create metadata for synthesis
            from src.core.models import PaperMetadata
            paper_metadata = PaperMetadata(
                arxiv_id=paper['arxiv_id'],
                title=paper['title'],
                authors=paper['authors'],
                abstract=paper.get('abstract', ''),
                categories=paper.get('categories', []),
                published_date=paper.get('published', ''),
                pdf_url=paper.get('pdf_url', '')
            )
            
            # Run Reduce phase to get Ochiai format
            ochiai_result = analyzer.synthesize_to_ochiai_format(
                section_analyses, 
                paper_metadata,
                state.get('query', '')
            )
            
            # Calculate approximate tokens used
            tokens_used = len(paper_memory.full_text) // 4  # Rough estimate
            
            analyzed_paper = {
                "metadata": paper,
                "analysis": ochiai_result.model_dump() if ochiai_result else {},
                "analysis_type": "gemini_map_reduce",
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
