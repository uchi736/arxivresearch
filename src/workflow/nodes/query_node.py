"""
Query generation node implementation
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
from src.core.agent_evaluation_keywords import AgentEvaluationKeywordGenerator
from .base import (
    logger,
    get_progress_tracker, save_progress_tracker, paper_memories
)


def generate_queries_node(state: AdvancedAgentState, container: AppContainer):
    """Generate search queries based on research plan with multilingual support"""
    logger.info("--- 検索クエリを生成中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("検索クエリ生成")
    
    try:
        if state.get("research_plan") is None:
            raise ValueError("Research plan is None - likely an error in plan_research_advanced_node")
        
        # Check if we have improved plan with better keywords
        if state.get("improved_research_plan"):
            logger.info("Using improved research plan for query generation")
            improved_plan = ImprovedResearchPlan.parse_obj(state["improved_research_plan"])
            
            # Check if this is an agent evaluation query and enhance keywords
            agent_eval_generator = AgentEvaluationKeywordGenerator()
            if agent_eval_generator.is_agent_evaluation_query(improved_plan.original_query):
                logger.info("Detected agent evaluation query - enhancing keywords")
                enhanced_keywords = agent_eval_generator.enhance_existing_keywords(improved_plan.search_keywords)
                improved_plan.search_keywords = enhanced_keywords
            
            # Build queries from improved plan's search keywords
            queries = []
            
            # Calculate papers per category
            total_papers = improved_plan.num_papers
            keyword_categories = [cat for cat, kws in improved_plan.search_keywords.items() if kws]
            papers_per_category = max(1, total_papers // len(keyword_categories)) if keyword_categories else total_papers
            
            # Generate queries from categorized keywords
            for category, keywords in improved_plan.search_keywords.items():
                if keywords:
                    # Use translated query for English terms
                    if improved_plan.query_language == "ja" and improved_plan.translated_query != improved_plan.original_query:
                        # Combine translated main term with specific keywords
                        query_keywords = keywords[:2]  # Limit to 2 keywords
                    else:
                        query_keywords = keywords[:3]  # Can use more keywords
                    
                    # Determine sort strategy based on time range
                    sort_by = "relevance"
                    if improved_plan.time_range == "recent":
                        sort_by = "lastUpdatedDate"
                    elif improved_plan.time_range == "foundational":
                        sort_by = "relevance"  # Still use relevance but will get cited papers
                    
                    # Add arXiv category if available
                    category_param = improved_plan.arxiv_categories[0] if improved_plan.arxiv_categories else None
                    
                    queries.append(SearchQuery(
                        keywords=query_keywords,
                        max_results=papers_per_category,
                        sort_by=sort_by,
                        category=category_param
                    ))
            
            # Add synonym-based queries if we have room
            remaining_papers = total_papers - sum(q.max_results for q in queries)
            if remaining_papers > 0 and improved_plan.synonyms:
                for term, synonyms in improved_plan.synonyms.items():
                    if remaining_papers <= 0:
                        break
                    if synonyms:
                        queries.append(SearchQuery(
                            keywords=[term, synonyms[0]],
                            max_results=min(3, remaining_papers),
                            sort_by="relevance"
                        ))
                        remaining_papers -= min(3, remaining_papers)
            
            logger.info(f"Generated {len(queries)} queries from improved plan")
            
            # Complete the step
            tracker.complete_step("検索クエリ生成", {
                "total_queries": len(queries),
                "topics": [q.keywords[0] if q.keywords else "" for q in queries],
                "from_improved_plan": True
            })
            
            return {
                "search_queries": [q.model_dump() for q in queries],
                "progress_tracker": save_progress_tracker(tracker)
            }
        
        # Fallback to original logic
        plan = ResearchPlan.parse_obj(state["research_plan"])
        
        generate_queries_prompt = ChatPromptTemplate.from_template(
            """以下の調査計画に基づいて、arXiv検索用のクエリをJSON形式で生成してください。

調査計画:
- メイントピック: {main_topic}
- サブトピック: {sub_topics}
- 注目分野: {focus_areas}

各トピックに対して効果的な検索クエリを作成してください。
重要: 
- キーワードは2-3個に限定し、シンプルにしてください
- 全クエリの合計で{num_papers}件以内の論文を取得するよう、max_resultsを調整してください
- あまり具体的すぎないキーワードを使用してください

出力は以下のJSON形式に従ってください:
{{
  "queries": [
    {{
      "keywords": ["keyword1", "keyword2"],
      "max_results": 1,
      "sort_by": "relevance",
      "category": "cs.AI"
    }}
  ]
}}
"""
        )
        
        response = container.llm_model.invoke(
            generate_queries_prompt.format(
                main_topic=plan.main_topic,
                sub_topics=", ".join(plan.sub_topics),
                focus_areas=", ".join(plan.focus_areas),
                num_papers=plan.num_papers
            )
        )
        
        # Extract JSON from response
        json_str = response.content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        
        queries_data = json.loads(json_str)
        queries = [SearchQuery.parse_obj(q) for q in queries_data["queries"]]
        
        # Complete the step
        tracker.complete_step("検索クエリ生成", {
            "total_queries": len(queries),
            "topics": [q.keywords[0] if q.keywords else "" for q in queries]
        })
        
        return {
            "search_queries": [q.model_dump() for q in queries],
            "progress_tracker": save_progress_tracker(tracker)
        }
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"  クエリ生成のJSONパースに失敗: {e}")
        # Fallback to simple query
        queries = [{
            "keywords": [plan.main_topic], 
            "max_results": 10, 
            "sort_by": "relevance", 
            "category": None
        }]
        
        tracker.complete_step("検索クエリ生成", {
            "total_queries": 1,
            "fallback": True,
            "error": str(e)
        })
        
        return {
            "search_queries": queries,
            "progress_tracker": save_progress_tracker(tracker)
        }
    
    except Exception as e:
        import traceback
        logger.debug(f"\n--- EXCEPTION in generate_queries_node ---")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.debug(f"Research plan: {state.get('research_plan', 'N/A')}")
        logger.debug("\nFull traceback:")
        traceback.print_exc()
        logger.debug("--- END EXCEPTION ---\n")
        
        # Create minimal fallback query
        fallback_query = [{
            "keywords": [state.get("initial_query", "research")], 
            "max_results": 10, 
            "sort_by": "relevance", 
            "category": None
        }]
        
        tracker.error_step("検索クエリ生成", str(e))
        
        return {
            "search_queries": fallback_query,
            "progress_tracker": save_progress_tracker(tracker)
        }
