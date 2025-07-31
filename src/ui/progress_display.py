"""
Progress display components for Streamlit UI

This module provides functions to display workflow progress in Streamlit.
"""

import streamlit as st
import time
from typing import Dict, Optional
from src.core.progress_tracker import ProgressTracker, StepStatus


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}時間"


def get_status_icon(status: StepStatus) -> str:
    """Get icon for step status"""
    icons = {
        StepStatus.PENDING: "[待機]",
        StepStatus.IN_PROGRESS: "[実行中]",
        StepStatus.COMPLETED: "[完了]",
        StepStatus.ERROR: "[エラー]",
        StepStatus.SKIPPED: "[スキップ]"
    }
    return icons.get(status, "[?]")


def display_progress(tracker: ProgressTracker, container=None):
    """Display progress information in Streamlit"""
    if container is None:
        container = st.container()
    
    with container:
        # Overall progress header
        st.subheader("分析進捗状況")
        
        # Overall progress bar
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            progress = tracker.overall_progress / 100
            st.progress(progress, text=f"全体進捗: {tracker.overall_progress:.0f}%")
        
        with col2:
            elapsed = format_time(tracker.elapsed_time)
            st.metric("経過時間", elapsed)
        
        with col3:
            remaining = tracker.estimate_remaining_time()
            if remaining:
                st.metric("推定残り時間", format_time(remaining))
            else:
                st.metric("推定残り時間", "計算中...")
        
        # Current step info
        current = tracker.current_step
        if current:
            st.info(f"現在の処理: {current.name}")
            if current.current_item:
                st.caption(f"詳細: {current.current_item}")
            
            # Show progress for current step if applicable
            if current.total_items > 0:
                step_progress = current.progress_percentage / 100
                st.progress(step_progress, 
                          text=f"{current.name}: {current.completed_items}/{current.total_items} ({current.progress_percentage:.0f}%)")
        
        # Steps list
        st.markdown("### 処理ステップ")
        
        for step_name in tracker.WORKFLOW_STEPS:
            step = tracker.steps[step_name]
            icon = get_status_icon(step.status)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                if step.status == StepStatus.IN_PROGRESS:
                    st.markdown(f"**{icon} {step_name}** *(処理中)*")
                else:
                    st.markdown(f"{icon} {step_name}")
            
            with col2:
                if step.duration > 0:
                    st.caption(format_time(step.duration))
            
            with col3:
                if step.status == StepStatus.ERROR and step.error_message:
                    st.error(step.error_message)
                elif step.status == StepStatus.IN_PROGRESS and step.total_items > 0:
                    st.caption(f"{step.completed_items}/{step.total_items}")
                elif step.status == StepStatus.COMPLETED and step.details:
                    # Show relevant details
                    if "total_papers" in step.details:
                        st.caption(f"{step.details['total_papers']}件")
                    elif "total_queries" in step.details:
                        st.caption(f"{step.details['total_queries']}個")
                    elif "analyzed_papers" in step.details:
                        st.caption(f"{step.details['analyzed_papers']}件")
        
        # Error or completion message
        if tracker.error_occurred:
            st.error("エラーが発生しました。詳細は上記をご確認ください。")
        elif tracker.is_complete:
            st.success("分析が完了しました！")
        
        # Tips
        with st.expander("ヒント", expanded=False):
            st.markdown("""
            - **論文翻訳**は最も時間がかかる処理です（1論文あたり30-60秒）
            - **フルテキスト処理**はPDFのサイズによって処理時間が変わります
            - **論文分析**は分析の深さによって処理時間が大きく変わります
            - 処理中でもブラウザを閉じても大丈夫です（バックグラウンドで継続されます）
            """)


def display_progress_async(progress_container, state_getter, interval=0.5):
    """
    Display progress asynchronously with periodic updates
    
    Args:
        progress_container: Streamlit container for progress display
        state_getter: Function to get current state with progress_tracker
        interval: Update interval in seconds
    """
    while True:
        state = state_getter()
        if state and state.get("progress_tracker"):
            tracker_data = state["progress_tracker"]
            tracker = ProgressTracker()
            
            # Reconstruct tracker from data
            tracker.start_time = tracker_data.get("start_time", tracker.start_time)
            tracker.is_complete = tracker_data.get("is_complete", False)
            tracker.error_occurred = tracker_data.get("error_occurred", False)
            
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
            
            # Clear and redraw
            progress_container.empty()
            with progress_container.container():
                display_progress(tracker)
            
            # Check if complete
            if tracker.is_complete or tracker.error_occurred:
                break
        
        time.sleep(interval)