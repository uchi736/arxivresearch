"""
Report generation node implementation
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


def generate_advanced_report_node(state: AdvancedAgentState, container: AppContainer):
    """Generate advanced report node"""
    logger.info("--- 高度なレポートを生成中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("レポート生成")
    
    # Build final report
    final_report = f"""# arXiv論文調査レポート（全文解析版）

## 調査概要
- **クエリ**: {state['initial_query']}
- **分析モード**: {state['analysis_mode']}
- **分析論文数**: {len(state['analyzed_papers'])}
- **総トークン使用量**: {sum(p.get('tokens_used', 0) for p in state['analyzed_papers'])}
"""
    
    # Add detailed information for each paper
    for paper in state["analyzed_papers"]:
        metadata = paper["metadata"]
        analysis = paper["analysis"]
        
        final_report += f"""
### {metadata['title']}
- **著者**: {', '.join(metadata['authors'])}
- **arXiv**: [{metadata['arxiv_id']}]({metadata['pdf_url']})

#### 落合フォーマット分析

**1. これは何か？**
{analysis.get('what_is_it', 'N/A')}

**2. 先行研究との比較**
{analysis.get('comparison_with_prior_work', 'N/A')}

**3. 技術の核心**
{analysis.get('key_technique', 'N/A')}

**4. 検証方法**
{analysis.get('validation_method', 'N/A')}

**5. 実験結果**
{analysis.get('experimental_results', 'N/A')}

**6. 議論点**
{analysis.get('discussion_points', 'N/A')}

**7. 実装詳細**
{analysis.get('implementation_details', 'N/A')}

**8. なぜ選ばれたか**
{analysis.get('why_selected', 'N/A')}

**9. 応用可能性**
{analysis.get('applicability', 'N/A')}

**10. 次に読むべき論文**
{', '.join(analysis.get('next_papers', [])) if analysis.get('next_papers') else 'N/A'}

---
"""
    
    # Complete the step
    tracker.complete_step("レポート生成", {
        "report_length": len(final_report)
    })
    
    return {
        "final_report": final_report,
        "progress_tracker": save_progress_tracker(tracker)
    }
