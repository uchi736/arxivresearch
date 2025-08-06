"""
Research planning node implementation
"""

import time
import traceback
from src.core.models import AdvancedAgentState, ResearchPlan
from src.core.research_planner import ResearchPlanner
from src.core.config import get_model_config, get_analysis_config
from src.core.dependencies import AppContainer
from .base import (
    logger,
    get_progress_tracker, save_progress_tracker
)


def plan_research_advanced_node(state: AdvancedAgentState, container: AppContainer):
    """Advanced research planning node with improved multilingual support"""
    logger.info("--- 高度な調査計画を策定中 ---")
    logger.debug(f"[PLANNING] Starting at {time.strftime('%H:%M:%S')}")
    
    # Get current model config
    model_config = container.config.model
    logger.debug(f"Model config - use_vertex_ai: {model_config.use_vertex_ai}")
    logger.debug(f"Model name: {model_config.model_name}")
    logger.debug(f"Location: {model_config.vertex_ai_location}")
    logger.debug(f"Project: {model_config.vertex_ai_project}")
    
    # The container will now manage the model instance, so manual reset is not needed here.
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("研究計画策定")
    
    try:
        logger.debug("Using new ResearchPlanner...")
        
        # Create research planner with the model from the container
        planner = ResearchPlanner(model=container.llm_model)
        
        # Get analysis mode from state
        analysis_mode = state.get("analysis_mode", "advanced_moderate")
        
        # Create improved research plan
        invoke_start = time.time()
        try:
            improved_plan = planner.create_research_plan_sync(
                state["initial_query"], 
                analysis_mode
            )
            invoke_time = time.time() - invoke_start
            logger.info(f"Research plan created in {invoke_time:.1f}s")
            
            # Log plan details
            logger.info(f"Original query: {improved_plan.original_query}")
            logger.info(f"Translated query: {improved_plan.translated_query}")
            logger.info(f"Language: {improved_plan.query_language}")
            logger.info(f"Number of papers: {improved_plan.num_papers}")
            logger.info(f"Analysis depth: {improved_plan.analysis_depth}")
            logger.info(f"Time range: {improved_plan.time_range}")
            
        except Exception as e:
            invoke_time = time.time() - invoke_start
            logger.error(f"Research plan creation failed after {invoke_time:.1f}s: {e}")
            raise
        
        # Convert to legacy format for compatibility
        plan = planner.convert_to_legacy_plan(improved_plan)
        
        # Set token budget based on analysis depth
        analysis_config = container.config.analysis
        if plan.analysis_depth == "shallow":
            token_budget = analysis_config.token_budget_shallow
        elif plan.analysis_depth == "moderate":
            token_budget = analysis_config.token_budget_moderate
        else:  # deep
            token_budget = analysis_config.token_budget_deep
        
        # Complete the step
        tracker.complete_step("研究計画策定", {
            "main_topic": plan.main_topic,
            "subtopics": len(plan.sub_topics),
            "analysis_depth": plan.analysis_depth,
            "language": improved_plan.query_language,
            "translated": improved_plan.translated_query != improved_plan.original_query
        })

        # Override num_papers if provided in the state
        if state.get("num_papers") is not None:
            logger.info(f"CLIで指定された論文数 ({state['num_papers']}) を使用します。")
            improved_plan.num_papers = state["num_papers"]
            plan.num_papers = state["num_papers"]
        
        # Store both plans for use in subsequent nodes
        return {
            "research_plan": plan.model_dump(),
            "improved_research_plan": improved_plan.model_dump(),  # Store improved plan
            "token_budget": token_budget,
            "progress_tracker": save_progress_tracker(tracker)
        }
    
    except Exception as e:
        logger.debug("\n--- EXCEPTION in plan_research_advanced_node ---")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.debug(f"Initial query: {state.get('initial_query', 'N/A')}")
        logger.debug("\nFull traceback:")
        traceback.print_exc()
        logger.debug("--- END EXCEPTION ---\n")
        
        tracker.error_step("研究計画策定", str(e))
        
        # Create a default plan to continue the workflow
        default_plan = ResearchPlan(
            main_topic=state.get("initial_query", "Research topic"),
            sub_topics=["General overview", "Related work", "Applications"],
            num_papers=10,
            focus_areas=["Key concepts", "Methods"],
            full_text_analysis=True,
            analysis_depth="moderate"
        )
        
        return {
            "research_plan": default_plan.model_dump(),
            "token_budget": state.get("token_budget", 30000),
            "progress_tracker": save_progress_tracker(tracker),
            "analysis_mode": "advanced_moderate"
        }
