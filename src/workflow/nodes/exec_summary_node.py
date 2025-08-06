"""
Executive Summary Generation Node
"""
import time
import os
from datetime import datetime, timezone
from typing import Dict, List
import json

from src.core.models import AdvancedAgentState, ExecSummary, OchiaiFormatAdvanced
from src.core.dependencies import AppContainer
from .base import logger, get_progress_tracker, save_progress_tracker

class PromptTemplateManager:
    """Manages loading of prompt templates."""
    def __init__(self, base_path="src/core/prompts"):
        self.base_path = base_path

    def get_template(self, template_name: str, lang: str = "ja", version: str = "latest") -> str:
        """Loads a specific prompt template."""
        file_path = os.path.join(self.base_path, template_name, lang, f"{version}.prompt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt template not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

class ExecSummaryNode:
    """
    Node to generate an executive summary from multiple paper analyses.
    """
    def __init__(self, container: AppContainer):
        self.llm_client = container.creative_llm_model  # Use the creative model for summarization
        self.prompt_manager = PromptTemplateManager()

    def run(self, state: AdvancedAgentState) -> Dict:
        """
        Generates and validates the executive summary.
        """
        logger.info("--- エグゼクティブサマリー生成を開始 ---")
        start_time = time.time()

        tracker = get_progress_tracker(state)
        tracker.start_step("サマリー生成")

        analysis_results = state.get("analyzed_papers", [])
        if not analysis_results:
            logger.warning("分析結果が見つからないため、サマリー生成をスキップします。")
            tracker.complete_step("サマリー生成", {"status": "skipped"})
            return {"exec_summary": None, "progress_tracker": save_progress_tracker(tracker)}

        try:
            # 1. Prepare data for prompt
            # NOTE: OchiaiFormatAdvanced is expected, but let's handle dicts for now
            summaries_for_prompt = []
            for paper in analysis_results:
                # The actual analysis content might be nested.
                # This part may need adjustment based on the final structure of `analyzed_papers`.
                analysis_data = paper.get("analysis", {})
                if isinstance(analysis_data, OchiaiFormatAdvanced):
                     summaries_for_prompt.append(analysis_data.model_dump())
                elif isinstance(analysis_data, dict): # Fallback for raw dicts
                     summaries_for_prompt.append(analysis_data)


            # 2. Load prompt and build final prompt
            # Try to load Claude-optimized prompt first, fallback to latest
            try:
                summary_prompt_template = self.prompt_manager.get_template("exec_summary", version="claude_optimized")
            except FileNotFoundError:
                logger.info("Claude最適化版プロンプトが見つからないため、標準版を使用します。")
                summary_prompt_template = self.prompt_manager.get_template("exec_summary")
            
            # Get query from state
            query = state.get("query", "")
            
            final_prompt = summary_prompt_template.format(
                analysis_data=json.dumps(summaries_for_prompt, indent=2, ensure_ascii=False),
                query=query
            )

            # 3. Call LLM
            logger.info("LLMにサマリー生成をリクエストします...")
            response = self.llm_client.invoke(final_prompt)
            summary_md = response.content

            # 4. Post-process and validate (simple validation for now)
            if not isinstance(summary_md, str) or not (800 <= len(summary_md) <= 1200):
                 logger.warning(f"生成されたサマリーの文字数が規定範囲外です。 (文字数: {len(summary_md)})")
                 # In a real scenario, we might retry or handle this error more gracefully.

            # 5. Create ExecSummary object
            exec_summary = ExecSummary(
                summary_md=summary_md,
                created_at=datetime.now(timezone.utc)
            )
            
            logger.info("エグゼクティブサマリーの生成が完了しました。")
            tracker.complete_step("サマリー生成", {"status": "success", "length": len(summary_md)})

        except Exception as e:
            logger.error(f"エグゼクティブサマリー生成中にエラーが発生しました: {e}", exc_info=True)
            tracker.complete_step("サマリー生成", {"status": "error", "error_message": str(e)})
            return {"exec_summary": None, "progress_tracker": save_progress_tracker(tracker)}

        end_time = time.time()
        logger.info(f"サマリー生成フェーズが {end_time - start_time:.2f}秒で完了しました。")

        return {
            "exec_summary": exec_summary,
            "progress_tracker": save_progress_tracker(tracker)
        }

def exec_summary_node(state: AdvancedAgentState, container: AppContainer) -> Dict:
    """Functional entry point for the exec summary node."""
    node = ExecSummaryNode(container)
    return node.run(state)
