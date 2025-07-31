import streamlit as st
import sys
import os
import json
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.core.workflow import build_advanced_workflow
from src.core.models import AdvancedAgentState
from src.core.progress_tracker import ProgressTracker, StepStatus
from src.ui.progress_display import display_progress

# Page config
st.set_page_config(
    page_title="arXiv Research Agent (Lightweight)",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = None

# Main UI
st.title("📚 arXiv Research Agent (軽量版)")
st.info("🚀 高速版：PDF翻訳機能を除いた軽量版です。検索・分析・レポート生成のみを実行します。")
st.divider()

# Research form
st.subheader("論文検索・分析")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "検索キーワード",
        placeholder="例: transformer architecture, reinforcement learning",
        help="調査したいトピックやキーワードを入力"
    )

with col2:
    analysis_depth = st.selectbox(
        "分析の深さ",
        ["shallow", "moderate", "deep"],
        index=1,
        format_func=lambda x: {
            "shallow": "簡易分析",
            "moderate": "標準分析", 
            "deep": "詳細分析"
        }[x]
    )

# Advanced options
with st.expander("詳細設定"):
    thread_id = st.text_input("スレッドID", value=f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    token_budget = st.number_input("トークン予算", min_value=10000, max_value=100000, value=30000)
    use_vertex_ai = st.checkbox("Vertex AIを使用", value=True, help="チェックを外すとGoogle AI APIを使用します")

# Search button
if st.button("検索・分析開始", type="primary"):
    if query:
        # Create placeholder for progress display
        progress_placeholder = st.empty()
        
        try:
            # Update config based on user selection
            from src.core.config import get_model_config
            model_config = get_model_config()
            model_config.use_vertex_ai = use_vertex_ai
            
            # Show configuration being used
            with progress_placeholder.container():
                st.info(f"検索・分析を開始しています... (API: {'Vertex AI' if use_vertex_ai else 'Google AI'})")
            
            # Build workflow
            try:
                arxiv_agent = build_advanced_workflow()
            except Exception as e:
                st.error(f"ワークフローの初期化に失敗しました: {str(e)}")
                logger.error(f"Failed to build workflow: {e}")
                st.stop()
            
            # Initial state
            initial_state = {
                "initial_query": query,
                "research_plan": None,
                "search_queries": [],
                "found_papers": [],
                "analyzed_papers": [],
                "final_report": "",
                "token_budget": token_budget,
                "analysis_mode": f"advanced_{analysis_depth}",
                "total_tokens_used": 0,
                "progress_tracker": None
            }
            
            # Run research
            with progress_placeholder.container():
                st.info("ワークフローを実行中...")
                st.caption(f"API: {'Vertex AI' if use_vertex_ai else 'Google AI'} | モデル: {model_config.model_name}")
                st.warning("⏳ 初回実行時は10-20秒程度かかります。しばらくお待ちください...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Log before invoke
            logger.info(f"Starting workflow invoke with query: {query}")
            invoke_start_time = time.time()
            
            try:
                result = arxiv_agent.invoke(initial_state)
                invoke_time = time.time() - invoke_start_time
                logger.info(f"Workflow completed in {invoke_time:.1f} seconds")
            except Exception as e:
                invoke_time = time.time() - invoke_start_time
                logger.error(f"Workflow failed after {invoke_time:.1f} seconds: {e}")
                with progress_placeholder.container():
                    st.error(f"ワークフロー実行エラー: {str(e)}")
                    st.caption(f"実行時間: {invoke_time:.1f}秒")
                raise
            
            # Display final progress
            if result.get("progress_tracker"):
                tracker_data = result["progress_tracker"]
                tracker = ProgressTracker()
                
                # Reconstruct tracker from data
                tracker.start_time = tracker_data.get("start_time", tracker.start_time)
                tracker.is_complete = tracker_data.get("is_complete", False)
                tracker.error_occurred = tracker_data.get("error_occurred", False)
                
                # Restore step states
                for step_name, step_data in tracker_data.get("steps", {}).items():
                    if step_name in tracker.steps:
                        step = tracker.steps[step_name]
                        step.status = StepStatus(step_data.get("status", "pending"))
                        step.start_time = step_data.get("start_time")
                        step.end_time = step_data.get("end_time")
                        step.current_item = step_data.get("current_item")
                        step.total_items = step_data.get("total_items", 0)
                        step.completed_items = step_data.get("completed_items", 0)
                        step.error_message = step_data.get("error_message")
                        step.details = step_data.get("details", {})
                
                progress_placeholder.empty()
                with progress_placeholder.container():
                    display_progress(tracker)
            else:
                progress_placeholder.empty()
            
            # Store results
            st.session_state.research_results = result
            st.success("分析が完了しました！")
            
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"エラーが発生しました: {str(e)}")
    else:
        st.warning("検索キーワードを入力してください")

# Display results
if st.session_state.research_results:
    results = st.session_state.research_results
    
    st.divider()
    st.subheader("分析結果")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("分析論文数", len(results.get('analyzed_papers', [])))
    with col2:
        total_tokens = sum(p.get('tokens_used', 0) for p in results.get('analyzed_papers', []))
        st.metric("総トークン使用量", f"{total_tokens:,}")
    with col3:
        st.metric("検索クエリ数", len(results.get('search_queries', [])))
    
    # Final report
    if results.get('final_report'):
        st.subheader("📋 分析レポート")
        st.markdown(results['final_report'])
    
    # Papers
    st.subheader("📄 分析された論文")
    
    for i, paper in enumerate(results.get('analyzed_papers', [])):
        with st.expander(f"{i+1}. {paper['metadata']['title']}", expanded=False):
            # Metadata
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**著者**: {', '.join(paper['metadata']['authors'])}")
                st.write(f"**公開日**: {paper['metadata']['published']}")
            with col2:
                st.write(f"**カテゴリ**: {', '.join(paper['metadata']['categories'])}")
                coverage = paper.get('coverage', {})
                avg_coverage = sum(coverage.values()) / max(len(coverage), 1) * 100
                st.write(f"**カバレッジ**: {avg_coverage:.1f}%")
            
            # Summary
            st.write("**要約**")
            st.write(paper['metadata']['summary'])
            
            # Analysis
            if 'analysis' in paper:
                st.write("**分析結果**")
                st.json(paper['analysis'])
            
            # PDF link
            if 'pdf_url' in paper['metadata']:
                st.markdown(f"[📄 PDF を開く]({paper['metadata']['pdf_url']})")

# Footer
st.divider()
st.caption("arXiv Research Agent (軽量版) - 高速論文検索・分析ツール")

# Show restoration info
with st.expander("💡 完全版への復元方法"):
    st.markdown("""
    **PDF翻訳機能付き完全版に戻すには：**
    
    ```bash
    # ワークフローを復元
    cp backup/workflow_with_translation.py src/core/workflow.py
    
    # アプリを復元
    cp backup/app_with_translation.py app.py
    ```
    
    **軽量版の利点：**
    - ⚡ 処理速度が大幅向上
    - 💰 API使用量を削減
    - 🎯 検索・分析に特化
    """)