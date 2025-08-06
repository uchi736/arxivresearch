#!/usr/bin/env python3
"""
arXiv Research Workbench - User-Friendly Interface
直感的で使いやすい論文検索・管理UI
"""
import streamlit as st
import pandas as pd
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="arXiv Research Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, readable CSS
st.markdown("""
<style>
    /* Clean theme with good readability */
    .stApp {
        background-color: #fafafa;
    }
    
    /* Consistent font */
    * {
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f2937 !important;
        font-weight: 600;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 14px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 16px;
        min-height: 40px;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }
    
    /* Success button */
    .success-button > button {
        background-color: #059669 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Warning button */
    .warning-button > button {
        background-color: #d97706 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #059669;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        color: #065f46;
    }
    
    .error-box {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        color: #991b1b;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px !important;
    }
    
    /* Progress indicators */
    .progress-text {
        font-family: 'Consolas', monospace;
        background: #f3f4f6;
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    
    /* Remove default margins */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_results' not in st.session_state:
    st.session_state.last_results = []

class WorkbenchController:
    """Main controller for workbench operations"""
    
    @staticmethod
    def execute_search(query, mode='moderate', limit=10, skip_analyzed=True):
        """Execute search with user-friendly feedback"""
        if not query.strip():
            st.error("検索クエリを入力してください。")
            return False
        
        # Build command
        cmd = f"python cli_app.py search \"{query}\" --depth {mode} --papers {limit}"
        if skip_analyzed:
            cmd += " --skip-analyzed"
        
        # Execute with enhanced progress tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Show initial status
        status_placeholder.info(f"🚀 検索開始: {query}")
        progress_bar = progress_placeholder.progress(0)
        
        try:
            # Start the process
            import time
            start_time = time.time()
            
            with st.spinner(f"'{query}'を検索・分析中..."):
                # Show command for transparency
                with st.expander("実行コマンド"):
                    st.code(cmd, language="bash")
                
                # Simulate progress updates during execution
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, 
                    text=True, timeout=300  # 5 minute timeout
                )
                
                # Update progress to completion
                progress_bar.progress(100)
                elapsed_time = int(time.time() - start_time)
                
                if result.returncode == 0:
                    status_placeholder.success(f"✅ 検索完了: '{query}' ({elapsed_time}秒)")
                    
                    # Show results summary
                    if result.stdout:
                        # Extract useful info from stdout
                        lines = result.stdout.split('\n')
                        found_papers = [line for line in lines if 'Found' in line or '発見' in line]
                        analyzed_papers = [line for line in lines if 'Analyzed' in line or '分析' in line]
                        
                        summary_info = []
                        if found_papers:
                            summary_info.extend(found_papers[-2:])  # Last 2 found messages
                        if analyzed_papers:
                            summary_info.extend(analyzed_papers[-2:])  # Last 2 analysis messages
                        
                        if summary_info:
                            st.info("📊 " + " | ".join(summary_info))
                        
                        with st.expander("📋 詳細ログ"):
                            st.text(result.stdout)
                    
                    return True
                else:
                    st.error(f"❌ 検索エラー")
                    
                    # Show detailed error information
                    if result.stderr:
                        error_msg = result.stderr
                        if "No papers found" in error_msg:
                            st.warning("📭 指定した検索条件では論文が見つかりませんでした")
                            st.info("💡 ヒント: より一般的なキーワード（英語推奨）で試してください")
                        elif "timeout" in error_msg.lower():
                            st.warning("⏱️ 処理がタイムアウトしました")
                            st.info("💡 ヒント: 論文数を減らすか、shallowモードで試してください")
                        else:
                            with st.expander("🔍 詳細エラー情報"):
                                st.text(error_msg)
                    
                    return False
                    
        except subprocess.TimeoutExpired:
            st.error("⏱️ 処理がタイムアウトしました（5分制限）")
            return False
        except Exception as e:
            st.error(f"❌ エラー: {str(e)}")
            return False
    
    @staticmethod
    def export_database(filename=None):
        """Export database with feedback"""
        # Generate unique filename with timestamp
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.xlsx"
        
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        # Create exports directory if it doesn't exist
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        filepath = export_dir / filename
        
        cmd = f"python cli_app.py registry export --output {filepath}"
        
        with st.spinner("Excelファイルを作成中..."):
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    if filepath.exists():
                        st.success(f"✅ エクスポート完了: {filepath.name}")
                        # Return full path for download
                        return str(filepath)
                    else:
                        st.warning("⚠️ ファイルが見つかりません")
                        st.info(f"期待されたパス: {filepath}")
                        return None
                else:
                    st.error("❌ エクスポートエラー")
                    if result.stderr:
                        # More specific error handling
                        if "Permission denied" in result.stderr:
                            st.error("ファイルへのアクセス権限がありません。ファイルが開かれている可能性があります。")
                        else:
                            st.text(result.stderr)
                    return None
                    
            except subprocess.TimeoutExpired:
                st.error("⏱️ エクスポートがタイムアウトしました（30秒）")
                return None
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                return None
    
    @staticmethod
    def translate_paper(arxiv_id):
        """Translate paper with feedback"""
        if not arxiv_id.strip():
            st.error("arXiv IDを入力してください。")
            return False
        
        # Convert arXiv ID to PDF URL if needed
        if not arxiv_id.startswith('http'):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            pdf_url = arxiv_id
        
        cmd = f"python cli_app.py translate \"{pdf_url}\" --academic"
        
        # Enhanced translation progress
        translation_placeholder = st.empty()
        translation_status = st.empty()
        
        translation_status.info(f"🌐 翻訳開始: {arxiv_id}")
        progress_bar = translation_placeholder.progress(0)
        
        with st.spinner(f"{arxiv_id}を翻訳中..."):
            start_time = time.time()
            
            try:
                # Show estimated time
                st.info("⏱️ 予想処理時間: 2-3分")
                
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, 
                    text=True, timeout=180  # 3 minute timeout
                )
                
                progress_bar.progress(100)
                elapsed_time = int(time.time() - start_time)
                
                if result.returncode == 0:
                    translation_status.success(f"✅ 翻訳完了: {arxiv_id} ({elapsed_time}秒)")
                    
                    # Save translation
                    translations_dir = Path("translations")
                    translations_dir.mkdir(exist_ok=True)
                    
                    safe_filename = arxiv_id.replace('/', '_').replace(':', '_').replace('?', '_')
                    output_path = translations_dir / f"{safe_filename}.html"
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    
                    st.info(f"💾 保存先: {output_path}")
                    return True
                else:
                    st.error("❌ 翻訳エラー")
                    st.text(result.stderr)
                    return False
                    
            except subprocess.TimeoutExpired:
                st.error("⏱️ 翻訳がタイムアウトしました（3分制限）")
                return False
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                return False
    
    @staticmethod
    def load_database():
        """Load and return database"""
        try:
            # Ensure database directory exists
            db_path = Path('database')
            db_path.mkdir(exist_ok=True)
            
            csv_file = db_path / 'analyzed_papers.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file, encoding='utf-8')
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"📁 データベース読み込みエラー: {str(e)}")
            st.info("💡 検索を実行してデータベースを作成してください")
            return pd.DataFrame()
    
    @staticmethod
    def get_database_stats():
        """Get database statistics"""
        df = WorkbenchController.load_database()
        
        if df.empty:
            return {
                "total_papers": 0,
                "categories": 0,
                "recent_papers": 0,
                "latest_date": "なし"
            }
        
        stats = {
            "total_papers": len(df),
            "categories": 0,
            "recent_papers": 0,
            "latest_date": "なし"
        }
        
        # Categories
        if 'カテゴリ' in df.columns:
            categories = set()
            for cat_str in df['カテゴリ'].dropna():
                if isinstance(cat_str, str):
                    categories.update([c.strip() for c in cat_str.split('/')])
            stats["categories"] = len(categories)
        
        # Recent papers (2025)
        if '公開日' in df.columns and not df.empty:
            try:
                recent_mask = df['公開日'].str.startswith('2025', na=False)
                stats["recent_papers"] = recent_mask.sum()
                stats["latest_date"] = df['公開日'].max()
            except:
                stats["recent_papers"] = 0
                stats["latest_date"] = "エラー"
        
        return stats

def show_search_tab():
    """Search and Analysis tab"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    st.header("📝 論文検索・分析")
    st.markdown("arXivから論文を検索して詳細分析を行います")
    
    with st.form("search_form", clear_on_submit=False):
        # Search query input
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "🔍 検索キーワード",
                placeholder="例: GraphRAG, 深層学習, Transformer architecture",
                help="論文のタイトル、キーワード、または研究トピックを日本語・英語で入力"
            )
        
        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox(
                "分析の深さ",
                ["shallow", "moderate", "deep"],
                index=1,
                help="shallow=基本情報のみ, moderate=標準分析, deep=詳細分析（時間がかかる）"
            )
        
        with col2:
            limit = st.number_input(
                "最大論文数",
                min_value=1,
                max_value=50,
                value=10,
                help="分析する論文の最大数"
            )
        
        with col3:
            skip_analyzed = st.checkbox(
                "分析済みをスキップ",
                value=True,
                help="既に分析済みの論文を重複処理しない"
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 検索開始", type="primary", use_container_width=True)
    
    # Execute search
    if submitted:
        # Set processing flag
        st.session_state.processing = True
        
        if WorkbenchController.execute_search(query, mode, limit, skip_analyzed):
            st.session_state.processing = False
            st.rerun()  # Refresh to show new data
        else:
            st.session_state.processing = False
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Examples section
    with st.expander("💡 検索例とヒント"):
        st.markdown("""
        **検索例:**
        - `GraphRAG` - GraphRAG関連論文
        - `深層学習 最適化` - 深層学習の最適化手法
        - `large language model evaluation` - LLMの評価手法
        - `computer vision transformer` - Vision Transformer関連
        
        **ヒント:**
        - 英語での検索が推奨されます（論文の多くが英語のため）
        - 複数のキーワードを組み合わせると精度が向上します
        - 分析モードは初回は'moderate'がおすすめです
        - 処理時間: shallow(5分) < moderate(10分) < deep(20分)
        """)

def show_database_tab():
    """Database management tab"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    st.header("📊 論文データベース")
    
    # Load data
    df = WorkbenchController.load_database()
    stats = WorkbenchController.get_database_stats()
    
    # Stats display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 総論文数", stats["total_papers"])
    with col2:
        st.metric("🏷️ カテゴリ数", stats["categories"])
    with col3:
        st.metric("📅 2025年の論文", stats["recent_papers"])
    with col4:
        st.metric("🆕 最新", stats["latest_date"])
    
    if df.empty:
        st.markdown('<div class="info-box">📝 データベースはまだ空です。上の「論文検索・分析」タブで検索を実行してください。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Management buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        export_disabled = df.empty
        if st.button("📤 Excel出力", type="primary", help="全データをExcelファイルに出力" if not export_disabled else "データがありません", disabled=export_disabled):
            filename = WorkbenchController.export_database()
            if filename and os.path.exists(filename):
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="💾 ダウンロード",
                        data=f.read(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    with col2:
        if st.button("📈 統計表示", help="詳細統計を表示"):
            # Show actual statistics
            with st.expander("📊 データベース統計", expanded=True):
                if not df.empty:
                    st.subheader("基本統計")
                    col_stat1, col_stat2 = st.columns(2)
                    
                    with col_stat1:
                        st.metric("総論文数", len(df))
                        
                        # Category distribution
                        if 'カテゴリ' in df.columns:
                            categories = df['カテゴリ'].value_counts().head(5)
                            st.write("**カテゴリ分布 (Top 5):**")
                            for cat, count in categories.items():
                                st.write(f"- {cat}: {count}件")
                    
                    with col_stat2:
                        # Date range
                        if '公開日' in df.columns:
                            try:
                                dates = pd.to_datetime(df['公開日'], errors='coerce')
                                valid_dates = dates.dropna()
                                if not valid_dates.empty:
                                    st.metric("最古の論文", valid_dates.min().strftime('%Y-%m-%d'))
                                    st.metric("最新の論文", valid_dates.max().strftime('%Y-%m-%d'))
                            except:
                                st.write("日付情報が不正です")
                        
                        # Score distribution
                        if '重要度' in df.columns or 'relevance_score' in df.columns:
                            score_col = '重要度' if '重要度' in df.columns else 'relevance_score'
                            scores = pd.to_numeric(df[score_col], errors='coerce').dropna()
                            if not scores.empty:
                                st.metric("平均スコア", f"{scores.mean():.2f}")
                else:
                    st.info("データがありません")
    
    with col3:
        if st.button("💾 バックアップ", help="データベースをバックアップ"):
            cmd = "python cli_app.py registry backup"
            with st.spinner("バックアップ中..."):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ バックアップ完了")
                else:
                    st.error("❌ バックアップエラー")
    
    with col4:
        if st.button("⚠️ データベース削除", help="すべてのデータを削除（注意）"):
            # Use session state for confirmation dialog
            if 'confirm_delete' not in st.session_state:
                st.session_state.confirm_delete = False
            st.session_state.confirm_delete = True
    
    # Show confirmation dialog if delete button was clicked
    if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
        st.warning("⚠️ **データベース削除の確認**")
        st.error("この操作を実行すると、すべての分析済み論文データが削除されます。")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ 削除実行", type="primary"):
                # Execute database reset
                cmd = "python cli_app.py registry reset --force --no-backup"
                with st.spinner("データベースを削除中..."):
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ データベースを削除しました")
                        st.session_state.confirm_delete = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 削除エラー")
                        st.text(result.stderr)
        
        with col2:
            if st.button("❌ キャンセル"):
                st.session_state.confirm_delete = False
                st.rerun()
    
    # Data display
    st.subheader("論文一覧")
    
    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_text = st.text_input(
            "🔍 検索フィルタ", 
            placeholder="タイトル、著者、キーワードで絞り込み...",
            key="db_filter"
        )
    with col2:
        available_sort_cols = []
        for col in ['公開日', 'タイトル', '著者']:
            if col in df.columns:
                available_sort_cols.append(col)
        
        if available_sort_cols:
            sort_col = st.selectbox(
                "並び替え", 
                available_sort_cols, 
                key="db_sort"
            )
        else:
            sort_col = None
            st.text("並び替え: なし")
    with col3:
        show_count = st.selectbox("表示数", [10, 25, 50, 100], key="db_limit")
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_text:
        # Search in multiple columns
        search_cols = ['タイトル', '著者', 'カテゴリ']
        available_cols = [col for col in search_cols if col in filtered_df.columns]
        
        if available_cols:
            mask = pd.Series(False, index=filtered_df.index)
            for col in available_cols:
                mask |= filtered_df[col].astype(str).str.contains(search_text, case=False, na=False)
            filtered_df = filtered_df[mask]
    
    # Sort
    if sort_col and sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_col, ascending=False)
    
    # Display ALL 21 columns - show all available columns in database
    # First, check what columns actually exist in the dataframe
    existing_columns = list(filtered_df.columns)
    
    # Define all possible 21 columns we want to display
    all_21_columns = [
        'arxiv_id', 'タイトル', '著者', '公開日', '取得日', 'カテゴリ',
        'キーワード', '処理状態', '概要JP', '要点_一言', '新規性', 
        '手法', '実験設定', '結果', '考察', '今後の課題', 
        '応用アイデア', '重要度', 'リンク_pdf', '落合フォーマットURL', '備考'
    ]
    
    # Also check for alternative column names (English or variations)
    alternative_names = {
        'title': 'タイトル',
        'authors': '著者',
        'published_date': '公開日',
        'analyzed_at': '取得日',
        'categories': 'カテゴリ',
        'keywords': 'キーワード',
        'status': '処理状態',
        'analysis_summary': '概要JP',
        'summary': '概要JP',
        'one_line_summary': '要点_一言',
        'novelty': '新規性',
        'key_findings': '新規性',
        'methodology': '手法',
        'method': '手法',
        'experimental_setup': '実験設定',
        'validation_method': '実験設定',
        'results': '結果',
        'experimental_results': '結果',
        'discussion': '考察',
        'discussion_points': '考察',
        'future_work': '今後の課題',
        'limitations': '今後の課題',
        'application_ideas': '応用アイデア',
        'applicability': '応用アイデア',
        'importance': '重要度',
        'relevance_score': '重要度',
        'pdf_link': 'リンク_pdf',
        'pdf_url': 'リンク_pdf',
        'ochiai_url': '落合フォーマットURL',
        'ochiai_format_url': '落合フォーマットURL',
        'notes': '備考',
        'memo': '備考'
    }
    
    # Build the actual display columns
    display_cols = []
    display_names = {}  # Store display names for columns
    
    for desired_col in all_21_columns:
        # Check if column exists as-is
        if desired_col in existing_columns:
            display_cols.append(desired_col)
            display_names[desired_col] = desired_col
        else:
            # Check for alternative names
            found = False
            for alt_name, japanese_name in alternative_names.items():
                if alt_name in existing_columns and japanese_name == desired_col:
                    display_cols.append(alt_name)
                    display_names[alt_name] = desired_col
                    found = True
                    break
            
            # If still not found, add placeholder column with empty values
            if not found:
                # Add a new column with empty values for missing data
                filtered_df[desired_col] = ''
                display_cols.append(desired_col)
                display_names[desired_col] = desired_col
    
    if display_cols:
        # Prepare display dataframe with renamed columns
        display_df = filtered_df[display_cols].copy()
        
        # Rename columns to Japanese for display
        rename_dict = {}
        for col in display_cols:
            if col in display_names:
                rename_dict[col] = display_names[col]
        
        if rename_dict:
            display_df = display_df.rename(columns=rename_dict)
        
        # Ensure all 21 columns are present (reorder to match expected order)
        final_columns = []
        for col in all_21_columns:
            if col in display_df.columns:
                final_columns.append(col)
        
        # Display with horizontal scrolling for 21 columns
        st.dataframe(
            display_df[final_columns].head(show_count),
            use_container_width=True,
            height=400
        )
        
        st.info(f"📋 表示中: {min(show_count, len(filtered_df))} / {len(filtered_df)} 件（総数: {len(df)}件）| 21項目全表示")
    else:
        st.error("表示可能なデータがありません")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_translation_tab():
    """Translation tab"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    st.header("🌐 論文翻訳")
    st.markdown("arXiv論文を日本語に翻訳します（学術モード）")
    
    with st.form("translation_form"):
        # Input
        col1, col2 = st.columns([3, 1])
        with col1:
            arxiv_id = st.text_input(
                "📄 arXiv ID",
                placeholder="例: 2409.17580v1, 2503.13804v1",
                help="翻訳したい論文のarXiv IDを入力（URLからでもOK）"
            )
        
        with col2:
            academic_mode = st.checkbox(
                "学術モード",
                value=True,
                help="学術的な翻訳を行う（推奨）"
            )
        
        # Submit
        submitted = st.form_submit_button("🚀 翻訳実行", type="primary", use_container_width=True)
    
    # Execute translation
    if submitted:
        # Clean up arXiv ID if URL is provided
        clean_id = arxiv_id.strip()
        if "arxiv.org/abs/" in clean_id:
            clean_id = clean_id.split("arxiv.org/abs/")[-1]
        
        if WorkbenchController.translate_paper(clean_id):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show translation history
    st.subheader("📚 翻訳履歴")
    
    translations_dir = Path("translations")
    if translations_dir.exists() and translations_dir.is_dir():
        translation_files = list(translations_dir.glob("*.html"))
        
        if translation_files:
            # Sort by modification time
            translation_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for file_path in translation_files[:10]:  # Show last 10
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(f"📄 {file_path.stem}")
                
                with col2:
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    st.text(f"🕒 {mod_time.strftime('%m/%d %H:%M')}")
                
                with col3:
                    if st.button("👀 表示", key=f"view_{file_path.stem}"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            with st.expander(f"翻訳内容: {file_path.stem}", expanded=True):
                                st.markdown(content, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"ファイル読み込みエラー: {str(e)}")
        else:
            st.info("💡 翻訳履歴はまだありません。上のフォームから翻訳を実行してください。")
    else:
        st.info("💡 翻訳履歴はまだありません。")

def main():
    """Main application"""
    
    # Title
    st.title("📚 arXiv Research Workbench")
    st.markdown("**論文検索・分析・翻訳のための統合ツール**")
    
    # First time user guidance
    stats = WorkbenchController.get_database_stats()
    if stats["total_papers"] == 0:
        st.info("👋 初回利用ですか？「🔍 検索・分析」タブから論文検索を始めてください！")
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["🔍 検索・分析", "📊 データベース", "🌐 翻訳"])
    
    with tab1:
        show_search_tab()
    
    with tab2:
        show_database_tab()
    
    with tab3:
        show_translation_tab()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ システム情報")
        
        # System status with real-time info
        stats = WorkbenchController.get_database_stats()
        
        # Status indicators
        st.markdown("### 📊 現在の状況")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 分析済み", stats["total_papers"])
        with col2:
            # Check if processing is active
            if 'processing' in st.session_state and st.session_state.processing:
                st.markdown("🔄 **処理中**")
            else:
                st.markdown("✅ **待機中**")
        
        # Database health
        st.markdown("### 💾 データベース")
        if stats["total_papers"] > 0:
            st.success(f"正常 ({stats['total_papers']}件)")
            if stats["latest_date"] != "なし":
                st.caption(f"最新: {stats['latest_date']}")
        else:
            st.info("データなし")
        
        # Quick actions
        st.markdown("### ⚡ クイックアクション")
        if st.button("🔄 画面更新", key="sidebar_refresh"):
            st.rerun()
        
        if stats["total_papers"] > 0:
            if st.button("📤 即座にExport", key="sidebar_export"):
                filename = WorkbenchController.export_database()
                if filename:
                    st.success("Export完了!")
        
        # Quick help
        with st.expander("🔧 基本的な使い方"):
            st.markdown("""
            1. **検索・分析**タブで論文を検索
            2. **データベース**タブで結果を確認・エクスポート
            3. **翻訳**タブで論文を日本語化
            
            **初回利用の方へ:**
            - 検索から始めてください
            - 分析モードは'moderate'がおすすめ
            - 結果はExcelで出力できます
            """)
        
        # System info
        with st.expander("⚙️ システム情報"):
            st.markdown("""
            - **バックエンド**: Vertex AI (Gemini)
            - **処理時間**: 40秒程度/論文
            - **対応言語**: 日本語・英語
            - **出力形式**: Excel (21列詳細)
            """)
        
        # Warning
        st.warning("⚠️ Vertex AI使用中：処理に時間がかかる場合があります")

if __name__ == "__main__":
    main()
