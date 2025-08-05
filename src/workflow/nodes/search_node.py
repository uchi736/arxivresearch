"""
Paper search node implementation
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

from src.search.relevance_scorer import (
    RelevanceScorer, remove_duplicates_smart, filter_by_time_range, filter_quality
)
from src.search.arxiv_search import search_arxiv_papers



def search_papers_node(state: AdvancedAgentState, container: AppContainer):
    """Enhanced search papers node with relevance scoring"""
    logger.info("--- arXivで論文を検索中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("論文検索", {
        "total_items": len(state["search_queries"])
    })
    
    all_papers = []
    
    for i, query_dict in enumerate(state["search_queries"]):
        try:
            query = SearchQuery.parse_obj(query_dict)
            query_str = ' '.join(query.keywords)
            logger.debug(f"検索中: {query_str}")
            
            # Update progress
            tracker.update_step("論文検索", 
                              f"検索中: {query_str}",
                              {"completed_items": i, "found": len(all_papers)})
            
            papers = search_arxiv_papers(query)
            all_papers.extend(papers)
            logger.debug(f"  → {len(papers)}件の論文を取得")
        except Exception as e:
            logger.debug(f"  → エラー: {e}")
            continue
    
    logger.debug(f"\n--- 検索結果の処理中 ---")
    logger.debug(f"取得論文総数: {len(all_papers)}件")
    
    # Get improved plan if available
    improved_plan = None
    if state.get("improved_research_plan"):
        improved_plan = ImprovedResearchPlan.parse_obj(state["improved_research_plan"])
        logger.debug(f"改善された研究計画を使用: time_range={improved_plan.time_range}")
    
    # 1. Quality filtering
    quality_papers = filter_quality(all_papers)
    logger.debug(f"品質フィルタ後: {len(quality_papers)}件")
    
    # 2. Time range filtering (if improved plan available)
    if improved_plan:
        time_filtered = filter_by_time_range(quality_papers, improved_plan.time_range)
        logger.debug(f"時間範囲フィルタ後: {len(time_filtered)}件")
    else:
        time_filtered = quality_papers
    
    # 3. Smart deduplication
    unique_papers = remove_duplicates_smart(time_filtered)
    logger.debug(f"重複除去後: {len(unique_papers)}件")
    
    # 4. Relevance scoring (if improved plan available)
    if improved_plan:
        logger.debug("\n--- 関連性スコアリング中 ---")
        scorer = container.relevance_scorer
        
        # Calculate scores for all papers
        scored_papers = []
        for paper in unique_papers:
            score, details = scorer.calculate_score(paper, improved_plan)
            # Add score as attribute for sorting
            paper_dict = paper.model_dump()
            paper_dict['relevance_score'] = score
            paper_dict['score_details'] = details
            scored_papers.append((score, paper_dict))
        
        # Sort by score (descending)
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Show top scores
        logger.info("\nトップ5論文のスコア:")
        for i, (score, paper) in enumerate(scored_papers[:5]):
            logger.info(f"  {i+1}. {paper['title'][:60]}... (score: {score:.2f})")
            if 'score_details' in paper:
                details = paper['score_details']
                logger.debug(f"     詳細: keyword={details.get('keyword', 0):.1f}, "
                      f"category={details.get('category', 0):.1f}, "
                      f"temporal={details.get('temporal', 0):.1f}")
        
        # Extract papers from tuples
        unique_papers = [paper for _, paper in scored_papers]
    else:
        # Convert to dict format without scoring
        unique_papers = [p.model_dump() for p in unique_papers]
    
    # 5. Limit to target number
    target_count = 10  # Default
    if state.get("research_plan"):
        plan = ResearchPlan.parse_obj(state["research_plan"])
        target_count = plan.num_papers
    elif improved_plan:
        target_count = improved_plan.num_papers
    
    if len(unique_papers) > target_count:
        logger.debug(f"\n取得論文数を{target_count}件に制限します")
        unique_papers = unique_papers[:target_count]
    
    # Complete the step
    tracker.complete_step("論文検索", {
        "total_papers": len(unique_papers),
        "queries_processed": len(state["search_queries"]),
        "used_scoring": improved_plan is not None
    })
    
    logger.debug(f"\n--- {len(unique_papers)}件の論文を選定完了 ---")
    return {
        "found_papers": unique_papers,
        "progress_tracker": save_progress_tracker(tracker)
    }
