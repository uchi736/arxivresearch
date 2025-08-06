"""
Data models for CSV paper registry system
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class AnalyzedPaper:
    """Analyzed paper record for CSV storage"""
    # Basic information
    arxiv_id: str
    title: str
    authors: str  # Semicolon-separated
    analyzed_at: str  # ISO format
    query_used: str
    relevance_score: float
    status: str  # completed/pending/failed
    analysis_summary: str
    categories: str  # Semicolon-separated
    
    # Analysis details (Ochiai format)
    what_is_it: str = ""
    key_technique: str = ""
    experimental_results: str = ""
    discussion_points: str = ""
    validation_method: str = ""
    implementation_details: str = ""
    
    # Metadata
    abstract: str = ""
    published_date: str = ""
    pdf_url: str = ""
    analysis_mode: str = "moderate"
    tokens_used: int = 0
    next_papers: str = ""  # Semicolon-separated suggestions


@dataclass
class SearchHistory:
    """Search history record for CSV storage"""
    query: str
    executed_at: str  # ISO format
    papers_found: int
    papers_new: int
    papers_skipped: int
    papers_analyzed: int
    execution_time: int  # seconds
    analysis_mode: str


class PaperRegistryConfig(BaseModel):
    """Configuration for CSV paper registry"""
    db_path: str = Field(default="database/", description="Path to CSV database files")
    papers_file: str = Field(default="analyzed_papers.csv", description="Papers CSV filename")
    history_file: str = Field(default="search_history.csv", description="History CSV filename")
    max_history_days: int = Field(default=90, description="Days to keep history")
    encoding: str = Field(default="utf-8-sig", description="CSV encoding (BOM for Excel)")
    schema_version: str = Field(default="full", description="Schema version: min, full, or legacy")
    
    @property
    def papers_path(self) -> str:
        return f"{self.db_path}/{self.papers_file}"
    
    @property
    def history_path(self) -> str:
        return f"{self.db_path}/{self.history_file}"
    
    @property 
    def columns(self) -> List[str]:
        """Get CSV columns based on schema version"""
        if self.schema_version == "min":
            return PAPERS_CSV_COLUMNS_MIN
        elif self.schema_version == "full":
            return PAPERS_CSV_COLUMNS_FULL
        elif self.schema_version == "legacy":
            return PAPERS_CSV_COLUMNS_LEGACY
        else:
            return PAPERS_CSV_COLUMNS_MIN  # Default to minimum


# New Japanese-focused CSV column definitions
# Minimum version (8 columns for initial survey)
PAPERS_CSV_COLUMNS_MIN = [
    "arxiv_id",        # arXiv ID（v番号込み）
    "タイトル",          # 論文タイトル
    "著者",            # 著者（; 区切り）
    "公開日",          # arXiv公開日（UTC）
    "カテゴリ",         # arXiv primary+secondary  
    "概要JP",          # 日本語概要（3-5行）
    "手法",            # 主な技術要素・アルゴリズム
    "結果"             # 定量結果のハイライト
]

# Full version (21 columns for detailed analysis)
PAPERS_CSV_COLUMNS_FULL = [
    # Basic information (8 columns)
    "arxiv_id",        # arXiv ID（v番号込み）
    "タイトル",          # 論文タイトル  
    "著者",            # 著者（; 区切り）
    "公開日",          # arXiv公開日（UTC）
    "取得日",          # このサマリーを作成した日時
    "カテゴリ",         # arXiv primary+secondary
    "キーワード",        # 独自に抽出した主要キーワード（, 区切り）
    "処理状態",         # todo / in_progress / done など
    
    # Analysis content (8 columns)
    "概要JP",          # 日本語概要（3-5行）
    "要点_一言",        # 140字以内の"刺さる"キャッチまとめ
    "新規性",          # 何が新しい？（箇条書き可）
    "手法",            # 主な技術要素・アルゴリズム
    "実験設定",         # データセット / 比較モデル / 評価指標
    "結果",            # 定量結果のハイライト
    "考察",            # 著者 or 自分の考察ポイント
    "今後の課題",       # リミテーション・次ステップ
    
    # Management (5 columns)
    "応用アイデア",      # 自分のプロジェクトに活かせそうな点
    "重要度",          # ★1–5（主観）
    "リンク_pdf",      # PDF直リンク
    "落合フォーマットURL", # 落合フォーマット分析レポートへのリンク
    "備考"             # その他メモ
]

# Legacy columns (for backward compatibility)
PAPERS_CSV_COLUMNS_LEGACY = [
    # Basic information (9 columns)
    "arxiv_id", "title", "authors", "analyzed_at", "query_used", 
    "relevance_score", "status", "analysis_summary", "categories",
    
    # Analysis details (6 columns) 
    "what_is_it", "key_technique", "experimental_results", 
    "discussion_points", "validation_method", "implementation_details",
    
    # Metadata (6 columns)
    "abstract", "published_date", "pdf_url", 
    "analysis_mode", "tokens_used", "next_papers"
]

# Default columns (use minimum for lightweight operation)
PAPERS_CSV_COLUMNS = PAPERS_CSV_COLUMNS_MIN

HISTORY_CSV_COLUMNS = [
    "query", "executed_at", "papers_found", "papers_new", 
    "papers_skipped", "papers_analyzed", "execution_time", "analysis_mode"
]

# Status constants
class PaperStatus:
    COMPLETED = "completed"
    PENDING = "pending" 
    FAILED = "failed"
    SKIPPED = "skipped"


# Japanese column mappings for display
JAPANESE_COLUMN_NAMES = {
    # Basic information
    "arxiv_id": "論文ID",
    "title": "タイトル", 
    "authors": "著者",
    "analyzed_at": "分析日時",
    "query_used": "検索クエリ",
    "relevance_score": "関連度スコア",
    "status": "ステータス",
    "analysis_summary": "分析概要",
    "categories": "カテゴリ",
    
    # Analysis details
    "what_is_it": "概要・内容",
    "key_technique": "主要技術", 
    "experimental_results": "実験結果",
    "discussion_points": "議論・考察",
    "validation_method": "検証方法",
    "implementation_details": "実装詳細",
    
    # Full format specific mappings
    "タイトル": "タイトル",
    "著者": "著者", 
    "公開日": "公開日",
    "取得日": "取得日",
    "カテゴリ": "カテゴリ",
    "キーワード": "キーワード",
    "処理状態": "処理状態",
    "概要JP": "概要JP",
    "要点_一言": "要点_一言",
    "新規性": "新規性",
    "手法": "手法",
    "実験設定": "実験設定", 
    "結果": "結果",
    "考察": "考察",
    "今後の課題": "今後の課題",
    "応用アイデア": "応用アイデア",
    "重要度": "重要度",
    "リンク_pdf": "リンク_pdf",
    "落合フォーマットURL": "落合フォーマットURL",
    "備考": "備考",
    
    # Metadata  
    "abstract": "アブストラクト",
    "published_date": "公開日",
    "pdf_url": "PDF URL",
    "analysis_mode": "分析モード",
    "tokens_used": "使用トークン数", 
    "next_papers": "関連論文候補"
}

# Status mappings to Japanese
JAPANESE_STATUS_NAMES = {
    "completed": "完了",
    "pending": "処理中",
    "failed": "失敗", 
    "skipped": "スキップ"
}