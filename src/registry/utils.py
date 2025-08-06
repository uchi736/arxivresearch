"""
Utility functions for CSV paper registry system
"""

import re
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalysisResultConverter:
    """Converts between analysis result formats and CSV rows"""
    
    @staticmethod
    def sanitize_csv_text(text: str) -> str:
        """
        Sanitize text for CSV storage
        - Replace newlines with \\n
        - Remove problematic characters
        - Truncate if too long
        """
        if not text:
            return ""
        
        # Replace newlines and tabs
        text = text.replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
        
        # Remove or replace problematic characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Truncate if too long (Excel cell limit is ~32K chars)
        if len(text) > 5000:  # Increased limit for better content preservation
            text = text[:4997] + "..."
        
        return text.strip()
    
    @staticmethod
    def join_list(items: List[str], separator: str = ";") -> str:
        """Join list items with separator, sanitizing each item"""
        if not items:
            return ""
        
        sanitized = [AnalysisResultConverter.sanitize_csv_text(str(item)) for item in items]
        return separator.join(filter(None, sanitized))
    
    @staticmethod
    def split_list(text: str, separator: str = ";") -> List[str]:
        """Split text back into list"""
        if not text:
            return []
        return [item.strip() for item in text.split(separator) if item.strip()]
    
    @staticmethod
    def json_to_csv_row(analysis_result: dict) -> dict:
        """
        Convert analysis result JSON to CSV row format
        
        Args:
            analysis_result: Full analysis result from workflow
            
        Returns:
            Dict with CSV column names as keys
        """
        # Extract metadata
        metadata = analysis_result.get('metadata', {})
        analysis = analysis_result.get('analysis', {})
        
        # Handle authors list
        authors = metadata.get('authors', [])
        authors_str = AnalysisResultConverter.join_list(authors)
        
        # Handle categories list  
        categories = metadata.get('categories', [])
        categories_str = AnalysisResultConverter.join_list(categories)
        
        # Handle next_papers list
        next_papers = analysis.get('next_papers', [])
        next_papers_str = AnalysisResultConverter.join_list(next_papers)
        
        # Current timestamp
        now = datetime.now().isoformat()
        
        return {
            # Basic information
            "arxiv_id": metadata.get('arxiv_id', ''),
            "title": AnalysisResultConverter.sanitize_csv_text(metadata.get('title', '')),
            "authors": authors_str,
            "analyzed_at": now,
            "query_used": AnalysisResultConverter.sanitize_csv_text(analysis_result.get('query_used', '')),
            "relevance_score": analysis_result.get('relevance_score', 0.0),
            "status": "completed",
            "analysis_summary": AnalysisResultConverter.sanitize_csv_text(analysis.get('what_is_it', '')[:200]),
            "categories": categories_str,
            
            # Analysis details (Ochiai format)
            "what_is_it": AnalysisResultConverter.sanitize_csv_text(analysis.get('what_is_it', '')),
            "key_technique": AnalysisResultConverter.sanitize_csv_text(analysis.get('key_technique', '')),
            "experimental_results": AnalysisResultConverter.sanitize_csv_text(analysis.get('experimental_results', '')),
            "discussion_points": AnalysisResultConverter.sanitize_csv_text(analysis.get('discussion_points', '')),
            "validation_method": AnalysisResultConverter.sanitize_csv_text(analysis.get('validation_method', '')),
            "implementation_details": AnalysisResultConverter.sanitize_csv_text(analysis.get('implementation_details', '')),
            
            # Metadata
            "abstract": AnalysisResultConverter.sanitize_csv_text(metadata.get('abstract', '')),
            "published_date": metadata.get('published_date', ''),
            "pdf_url": metadata.get('pdf_url', ''),
            "analysis_mode": analysis_result.get('analysis_mode', 'moderate'),
            "tokens_used": analysis_result.get('tokens_used', 0),
            "next_papers": next_papers_str
        }
    
    @staticmethod
    def csv_row_to_analysis(csv_row: dict) -> dict:
        """
        Convert CSV row back to analysis result format
        
        Args:
            csv_row: Dict from CSV with column names as keys
            
        Returns:
            Analysis result in original JSON format
        """
        # Restore newlines in text fields
        def restore_text(text: str) -> str:
            if not text:
                return ""
            return text.replace('\\n', '\n')
        
        # Split list fields
        authors = AnalysisResultConverter.split_list(csv_row.get('authors', ''))
        categories = AnalysisResultConverter.split_list(csv_row.get('categories', ''))
        next_papers = AnalysisResultConverter.split_list(csv_row.get('next_papers', ''))
        
        return {
            'metadata': {
                'arxiv_id': csv_row.get('arxiv_id', ''),
                'title': restore_text(csv_row.get('title', '')),
                'authors': authors,
                'abstract': restore_text(csv_row.get('abstract', '')),
                'published_date': csv_row.get('published_date', ''),
                'categories': categories,
                'pdf_url': csv_row.get('pdf_url', '')
            },
            'analysis': {
                'what_is_it': restore_text(csv_row.get('what_is_it', '')),
                'key_technique': restore_text(csv_row.get('key_technique', '')),
                'experimental_results': restore_text(csv_row.get('experimental_results', '')),
                'discussion_points': restore_text(csv_row.get('discussion_points', '')),
                'validation_method': restore_text(csv_row.get('validation_method', '')),
                'implementation_details': restore_text(csv_row.get('implementation_details', '')),
                'next_papers': next_papers
            },
            'query_used': csv_row.get('query_used', ''),
            'relevance_score': float(csv_row.get('relevance_score', 0.0)),
            'analysis_mode': csv_row.get('analysis_mode', 'moderate'),
            'tokens_used': int(csv_row.get('tokens_used', 0)),
            'analyzed_at': csv_row.get('analyzed_at', ''),
            'status': csv_row.get('status', 'completed')
        }


class AnalysisTranslator:
    """Translates English analysis results to Japanese"""
    
    # Simple translation mappings for common terms
    TRANSLATION_MAP = {
        # Technical terms
        "Large Language Model": "大規模言語モデル",
        "LLM": "LLM",
        "Machine Learning": "機械学習",
        "Deep Learning": "深層学習",
        "Neural Network": "ニューラルネットワーク",
        "Transformer": "Transformer",
        "Attention": "注意機構",
        "Dataset": "データセット",
        "Benchmark": "ベンチマーク",
        "API": "API",
        "Framework": "フレームワーク",
        "Algorithm": "アルゴリズム",
        "Performance": "性能",
        "Accuracy": "精度",
        "Evaluation": "評価",
        "Training": "訓練",
        "Fine-tuning": "ファインチューニング",
        "Inference": "推論",
        "Prompt": "プロンプト",
        "Embedding": "埋め込み",
        "Classification": "分類",
        "Regression": "回帰",
        "Clustering": "クラスタリング",
        "Optimization": "最適化",
        "Gradient": "勾配",
        "Loss Function": "損失関数",
        "Regularization": "正則化",
        "Overfitting": "過学習",
        "Generalization": "汎化",
        "Cross-validation": "交差検証",
        
        # Common phrases
        "This paper": "本論文",
        "The authors": "著者ら",
        "The proposed method": "提案手法",
        "The experiment": "実験",
        "The results": "結果",
        "shows that": "ことを示している",
        "demonstrates": "実証している",
        "achieves": "達成している",
        "outperforms": "上回っている",
        "compared to": "と比較して",
        "baseline": "ベースライン",
        "state-of-the-art": "最先端",
        "significantly": "大幅に",
        "improvement": "改善",
        "approach": "アプローチ",
        "methodology": "手法",
        "architecture": "アーキテクチャ",
        "implementation": "実装"
    }
    
    @staticmethod
    def translate_text(text: str) -> str:
        """
        Simple rule-based translation for technical content
        This is a fallback for when LLM translation is not available
        """
        if not text or len(text.strip()) == 0:
            return text
            
        # If text is already mostly Japanese, return as-is
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
        if japanese_chars > len(text) * 0.3:
            return text
            
        translated = text
        
        # Apply simple term translations
        for en_term, jp_term in AnalysisTranslator.TRANSLATION_MAP.items():
            translated = translated.replace(en_term, jp_term)
        
        return translated
    
    @staticmethod
    def translate_analysis_content(analysis_dict: dict) -> dict:
        """
        Translate the content fields of an analysis result
        
        Args:
            analysis_dict: Analysis dictionary with English content
            
        Returns:
            Analysis dictionary with translated content
        """
        translated = analysis_dict.copy()
        
        # Fields to translate
        translate_fields = [
            'what_is_it', 'key_technique', 'experimental_results', 
            'discussion_points', 'validation_method', 'implementation_details',
            'analysis_summary'
        ]
        
        for field in translate_fields:
            if field in translated and translated[field]:
                try:
                    translated[field] = AnalysisTranslator.translate_text(translated[field])
                except Exception as e:
                    logger.warning(f"Failed to translate field {field}: {e}")
                    # Keep original text if translation fails
                    pass
        
        return translated


class JapaneseAnalysisConverter:
    """Convert analysis results to Japanese-focused CSV format"""
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 5) -> str:
        """Extract key technical terms from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        common_terms = [
            "LLM", "Large Language Model", "Deep Learning", "Machine Learning",
            "Transformer", "Attention", "Neural Network", "API", "Dataset",
            "Benchmark", "Training", "Fine-tuning", "Embedding", "Classification",
            "GPT", "BERT", "CNN", "RNN", "AI", "NLP", "Computer Vision"
        ]
        
        found_keywords = []
        text_upper = text.upper()
        
        for term in common_terms:
            if term.upper() in text_upper and term not in found_keywords:
                found_keywords.append(term)
                if len(found_keywords) >= max_keywords:
                    break
        
        return ", ".join(found_keywords)
    
    @staticmethod
    def create_catchphrase(analysis: dict) -> str:
        """Create 140-char catchphrase from analysis"""
        what_is_it = analysis.get('what_is_it', '')
        key_tech = analysis.get('key_technique', '')
        
        # Simple combination and truncation
        catchphrase = f"{what_is_it[:50]}...{key_tech[:50]}"
        if len(catchphrase) > 140:
            catchphrase = catchphrase[:137] + "..."
        
        return catchphrase
    
    @staticmethod
    def to_japanese_min_format(analysis_result: dict) -> dict:
        """Convert to minimum 8-column Japanese format"""
        metadata = analysis_result.get('metadata', {})
        analysis = analysis_result.get('analysis', {})
        
        # Extract authors
        authors = metadata.get('authors', [])
        authors_str = AnalysisResultConverter.join_list(authors, "; ")
        
        # Extract categories  
        categories = metadata.get('categories', [])
        categories_str = AnalysisResultConverter.join_list(categories, " / ")
        
        return {
            "arxiv_id": metadata.get('arxiv_id', ''),
            "タイトル": AnalysisResultConverter.sanitize_csv_text(metadata.get('title', '')),
            "著者": authors_str,
            "公開日": metadata.get('published_date', '')[:10],  # YYYY-MM-DD format
            "カテゴリ": categories_str,
            "概要JP": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(analysis.get('what_is_it', ''))
            )[:300],  # 3-5 lines
            "手法": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(analysis.get('key_technique', ''))
            ),
            "結果": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(analysis.get('experimental_results', ''))
            )
        }
    
    @staticmethod
    def convert_importance_to_stars(relevance_score: float) -> str:
        """Convert relevance score to star rating (★1-5)"""
        if relevance_score >= 9.0:
            return "★★★★★"
        elif relevance_score >= 8.0:
            return "★★★★"
        elif relevance_score >= 6.0:
            return "★★★"
        elif relevance_score >= 4.0:
            return "★★"
        else:
            return "★"
    
    @staticmethod
    def generate_ochiai_report_url(arxiv_id: str) -> str:
        """Generate URL for Ochiai format analysis report"""
        # For now, create a placeholder URL structure
        # This could be enhanced to point to actual report generation system
        return f"https://reports.example.com/ochiai/{arxiv_id}"
    
    @staticmethod
    def to_japanese_full_format(analysis_result: dict) -> dict:
        """Convert to full 21-column Japanese format"""
        metadata = analysis_result.get('metadata', {})
        analysis = analysis_result.get('analysis', {})
        
        # Get minimum format as base
        base_row = JapaneseAnalysisConverter.to_japanese_min_format(analysis_result)
        
        # Extract full text for keyword extraction
        full_text = f"{analysis.get('what_is_it', '')} {analysis.get('key_technique', '')}"
        keywords = JapaneseAnalysisConverter.extract_keywords(full_text)
        
        # Create catchphrase
        catchphrase = JapaneseAnalysisConverter.create_catchphrase(analysis)
        
        # Convert relevance score to star rating
        relevance_score = analysis_result.get('relevance_score', 5.0)
        importance_stars = JapaneseAnalysisConverter.convert_importance_to_stars(relevance_score)
        
        # Generate Ochiai format URL
        arxiv_id = metadata.get('arxiv_id', '')
        ochiai_url = JapaneseAnalysisConverter.generate_ochiai_report_url(arxiv_id)
        
        # Add full format fields with detailed content
        full_row = base_row.copy()
        
        # Extract more detailed information from analysis
        comparison = analysis.get('comparison_with_prior_work', '')
        technique = analysis.get('key_technique', '')
        validation = analysis.get('validation_method', '')
        results = analysis.get('experimental_results', '')
        discussion = analysis.get('discussion_points', '')
        applicability = analysis.get('applicability', '')
        
        full_row.update({
            "取得日": datetime.now().strftime("%Y-%m-%d"),
            "キーワード": keywords,
            "処理状態": "done",
            
            # Enhanced analysis content with more detail
            "要点_一言": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(
                    analysis.get('what_is_it', catchphrase)[:200]
                )
            ),
            "新規性": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(comparison)[:2000]
            ),
            "手法": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(technique)[:2000]
            ),
            "実験設定": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(validation)[:2000]
            ),
            "結果": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(results)[:2000]
            ),
            "考察": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(discussion)[:2000]
            ),
            "今後の課題": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(
                    # Extract future work from discussion or use empty
                    "今後の課題：" + discussion.split("今後")[-1][:200] if "今後" in discussion else ""
                )
            ),
            
            # Application ideas from applicability field
            "応用アイデア": AnalysisTranslator.translate_text(
                AnalysisResultConverter.sanitize_csv_text(applicability)[:300]
            ),
            "重要度": importance_stars,
            "リンク_pdf": metadata.get('pdf_link', metadata.get('pdf_url', '')),
            "落合フォーマットURL": ochiai_url,
            "備考": f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        })
        
        return full_row