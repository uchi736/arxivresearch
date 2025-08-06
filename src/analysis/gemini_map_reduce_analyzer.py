"""
Gemini-based Map-Reduce analyzer for paper analysis

This module leverages Gemini's long context capabilities to analyze papers
using a Map-Reduce approach without complex RAG systems.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.core.models import OchiaiFormatAdvanced, PaperMetadata
from src.core.config import create_llm_model


class GeminiMapReduceAnalyzer:
    """Analyzer using Gemini's long context with Map-Reduce pattern"""
    
    def __init__(self, model=None):
        """Initialize analyzer with Gemini model"""
        self.model = model or create_llm_model()
        
        # Section analysis prompts (Map phase)
        self.SECTION_ANALYSIS_PROMPTS = {
            'abstract': """
論文の要約セクションを分析してください。

セクション内容:
{section_text}

以下の点を抽出してください：
1. 研究の主目的
2. 提案手法の概要
3. 主要な貢献
4. 結果の要約

JSON形式で回答してください。
""",
            
            'introduction': """
論文の導入セクションを分析してください。

セクション内容:
{section_text}

以下の点を抽出してください：
1. 研究背景と動機
2. 解決しようとする問題
3. 研究の新規性
4. 論文の構成

JSON形式で回答してください。
""",
            
            'methods': """
論文の手法セクションを分析してください。

セクション内容:
{section_text}

以下の点を抽出してください：
1. 提案手法の詳細
2. アルゴリズムやモデルの構造
3. 重要なパラメータ
4. 実装の詳細

JSON形式で回答してください。
""",
            
            'results': """
論文の結果セクションを分析してください。

セクション内容:
{section_text}

以下の点を抽出してください：
1. 主要な実験結果
2. ベースラインとの比較
3. 統計的有意性
4. 結果の解釈

JSON形式で回答してください。
""",
            
            'discussion': """
論文の議論セクションを分析してください。

セクション内容:
{section_text}

以下の点を抽出してください：
1. 結果の意義
2. 研究の限界
3. 今後の課題
4. 実用上の示唆

JSON形式で回答してください。
"""
        }
        
        # Ochiai format synthesis prompt (Reduce phase)
        self.OCHIAI_SYNTHESIS_PROMPT = """
以下の論文セクション分析結果を基に、落合陽一フォーマットで統合的な分析を作成してください。

論文タイトル: {title}
著者: {authors}

セクション分析結果:
{section_analyses}

以下の落合フォーマットで詳細に分析してください：

1. **what_is_it（これは何か？）**: 
   - この研究が何を扱っているか
   - 研究の核心的な問い

2. **comparison_with_prior_work（先行研究との比較）**:
   - 既存手法との違い
   - 改善点や新規性

3. **key_technique（技術の核心）**:
   - 提案手法の技術的詳細
   - アルゴリズムやモデルの革新的な点

4. **validation_method（検証方法）**:
   - 実験設定
   - 評価指標
   - データセット

5. **experimental_results（実験結果）**:
   - 定量的結果
   - 定性的結果
   - ベースラインとの比較

6. **discussion_points（議論点）**:
   - 結果の解釈
   - 限界と課題
   - 今後の展望

7. **implementation_details（実装詳細）**:
   - 重要な実装上の工夫
   - 再現可能性に関する情報

8. **why_selected（なぜ選ばれたか）**:
   - この論文の重要性
   - 検索クエリとの関連性

9. **applicability（応用可能性）**:
   - 実用的な応用先
   - 他分野への展開可能性

10. **next_papers（次に読むべき論文）**:
    - 参考文献から重要なもの
    - 関連研究の提案

JSON形式で、各項目を詳細に記述してください。
"""
        
        # Full paper analysis prompt (for shorter papers)
        self.FULL_PAPER_ANALYSIS_PROMPT = """
以下の論文全文を落合陽一フォーマットで分析してください。

論文タイトル: {title}
著者: {authors}

論文全文:
{full_text}

[以下、OCHIAI_SYNTHESIS_PROMPTと同じフォーマット指示]
"""
    
    def count_tokens(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters for Japanese/English mixed)"""
        return len(text) // 3  # Conservative estimate for mixed content

    def chunk_text(self, text: str, max_tokens: int = 25000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                
                # If single paragraph is too long, split by sentences
                if self.count_tokens(current_chunk) > max_tokens:
                    sentences = current_chunk.split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
                        if self.count_tokens(test_chunk) <= max_tokens:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def analyze_section(self, section_name: str, section_text: str) -> Dict:
        """
        Analyze a single section (Map phase) with token limit handling
        
        Args:
            section_name: Name of the section
            section_text: Text content of the section
            
        Returns:
            Analysis results as dictionary
        """
        # Skip if section is too short
        if len(section_text.strip()) < 50:
            return {"status": "skipped", "reason": "too_short"}
        
        # Check token count and chunk if necessary
        token_count = self.count_tokens(section_text)
        if token_count > 25000:  # Conservative limit for prompt + response
            print(f"  Large section detected ({token_count:,} tokens), chunking...")
            chunks = self.chunk_text(section_text, max_tokens=20000)
            
            # Analyze each chunk and combine results
            chunk_results = []
            for i, chunk in enumerate(chunks):
                print(f"    Analyzing chunk {i+1}/{len(chunks)}...")
                result = self._analyze_text_chunk(section_name, chunk)
                if result.get("status") != "skipped":
                    chunk_results.append(result)
            
            # Combine chunk results
            if not chunk_results:
                return {"status": "skipped", "reason": "all_chunks_failed"}
                
            return self._combine_chunk_results(section_name, chunk_results)
        
        # Normal analysis for smaller sections
        return self._analyze_text_chunk(section_name, section_text)
    
    def _analyze_text_chunk(self, section_name: str, section_text: str) -> Dict:
        """Analyze a single text chunk"""
        # Get appropriate prompt
        prompt_template = self.SECTION_ANALYSIS_PROMPTS.get(
            section_name, 
            self.SECTION_ANALYSIS_PROMPTS['abstract']  # Default
        )
        
        prompt = prompt_template.format(section_text=section_text)
        
        try:
            response = self.model.invoke(prompt)
            content = response.content.strip()
            
            # Try to parse JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            # Additional cleanup for common JSON issues
            content = content.strip()
            
            # Fix common unterminated string issues
            if content.count('"') % 2 != 0:
                # Odd number of quotes - likely unterminated string
                print(f"  Warning: Possible unterminated string in section '{section_name}', attempting to fix...")
                # Try to find the last complete JSON object
                try:
                    # Find the last complete closing brace
                    last_brace = content.rfind('}')
                    if last_brace > 0:
                        content = content[:last_brace + 1]
                except:
                    pass
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"  JSON parsing error in section '{section_name}': {e}")
            print(f"  Content around error: ...{content[max(0, e.pos-50):e.pos+50]}...")
            # Return a safe fallback structure
            return {
                "status": "json_error", 
                "error": f"JSON parsing failed: {str(e)}",
                "section": section_name,
                "fallback_content": content[:500] + "..." if len(content) > 500 else content
            }
        except Exception as e:
            print(f"  Error analyzing section '{section_name}': {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_response": response.content if 'response' in locals() else None
            }
    
    def _combine_chunk_results(self, section_name: str, chunk_results: List[Dict]) -> Dict:
        """Combine results from multiple chunks of the same section"""
        combined = {
            "status": "combined",
            "section_name": section_name,
            "chunk_count": len(chunk_results),
            "combined_analysis": {}
        }
        
        # Collect all keys from chunk results
        all_keys = set()
        for chunk in chunk_results:
            if isinstance(chunk, dict):
                all_keys.update(chunk.keys())
        
        # Remove meta keys
        meta_keys = {'status', 'error', 'section', 'fallback_content', 'raw_response'}
        analysis_keys = all_keys - meta_keys
        
        # Combine content for each key
        for key in analysis_keys:
            values = []
            for chunk in chunk_results:
                if key in chunk and chunk[key]:
                    values.append(str(chunk[key]))
            
            if values:
                # Join multiple values with appropriate separators
                if len(values) == 1:
                    combined["combined_analysis"][key] = values[0]
                else:
                    combined["combined_analysis"][key] = " | ".join(values)
        
        return combined
    
    def synthesize_to_ochiai_format(
        self, 
        section_analyses: Dict[str, Dict],
        paper_metadata: PaperMetadata,
        query: str
    ) -> OchiaiFormatAdvanced:
        """
        Synthesize section analyses into Ochiai format (Reduce phase)
        
        Args:
            section_analyses: Results from section analysis
            paper_metadata: Paper metadata
            query: Original search query
            
        Returns:
            OchiaiFormatAdvanced object
        """
        # Prepare section analyses text, filtering out error sections
        filtered_analyses = {}
        error_sections = []
        
        for section_name, analysis in section_analyses.items():
            if isinstance(analysis, dict) and analysis.get('status') in ['error', 'json_error']:
                error_sections.append(section_name)
                # Use fallback content if available
                if 'fallback_content' in analysis:
                    filtered_analyses[section_name] = {
                        'summary': analysis['fallback_content'][:200] + "...",
                        'status': 'partial'
                    }
            else:
                filtered_analyses[section_name] = analysis
        
        if error_sections:
            print(f"  Note: {len(error_sections)} sections had parsing errors: {', '.join(error_sections)}")
        
        analyses_text = json.dumps(filtered_analyses, ensure_ascii=False, indent=2)
        
        # Create synthesis prompt
        prompt = self.OCHIAI_SYNTHESIS_PROMPT.format(
            title=paper_metadata.title,
            authors=", ".join(paper_metadata.authors[:3]),  # Limit authors
            section_analyses=analyses_text
        )
        
        # Use structured analyzer for synthesis too
        from src.analysis.ochiai_structured_analyzer import OchiaiStructuredAnalyzer
        
        try:
            # Create a combined text from section analyses
            combined_text = f"""
セクション分析結果のサマリー:

{analyses_text}

論文タイトル: {paper_metadata.title}
検索クエリ: {query}
"""
            
            analyzer = OchiaiStructuredAnalyzer(self.model)
            return analyzer.analyze_with_validation(
                text=combined_text,
                paper_metadata=paper_metadata,
                query=query,
                max_retries=2
            )
            
        except Exception as e:
            print(f"  Error synthesizing Ochiai format: {e}")
            # Return minimal Ochiai format
            return self._create_fallback_ochiai(paper_metadata, query, str(e))
    
    def analyze_full_paper(
        self, 
        full_text: str,
        paper_metadata: PaperMetadata,
        query: str
    ) -> OchiaiFormatAdvanced:
        """
        Analyze full paper at once (for shorter papers)
        
        Args:
            full_text: Complete paper text
            paper_metadata: Paper metadata
            query: Original search query
            
        Returns:
            OchiaiFormatAdvanced object
        """
        prompt = self.FULL_PAPER_ANALYSIS_PROMPT.format(
            title=paper_metadata.title,
            authors=", ".join(paper_metadata.authors[:3]),
            full_text=full_text
        )
        
        # Add format instructions
        format_instructions = self.OCHIAI_SYNTHESIS_PROMPT.split("以下の落合フォーマットで")[1]
        # Ensure JSON output instruction is included
        if "JSON形式で" not in format_instructions:
            format_instructions += "\n\n必ずJSON形式で回答してください。"
        prompt = prompt.replace(
            "[以下、OCHIAI_SYNTHESIS_PROMPTと同じフォーマット指示]",
            format_instructions
        )
        
        # Use structured analyzer for more reliable output
        from src.analysis.ochiai_structured_analyzer import OchiaiStructuredAnalyzer
        
        try:
            analyzer = OchiaiStructuredAnalyzer(self.model)
            return analyzer.analyze_with_validation(
                text=full_text,
                paper_metadata=paper_metadata,
                query=query,
                max_retries=2
            )
        except Exception as e:
            print(f"  Error in full paper analysis: {e}")
            return self._create_fallback_ochiai(paper_metadata, query, str(e))
    
    def _create_fallback_ochiai(
        self, 
        paper_metadata: PaperMetadata, 
        query: str, 
        error_msg: str
    ) -> OchiaiFormatAdvanced:
        """Create fallback Ochiai format when analysis fails"""
        return OchiaiFormatAdvanced(
            what_is_it=f"タイトル: {paper_metadata.title}\n要約: {paper_metadata.abstract[:200]}...",
            comparison_with_prior_work="分析エラーのため取得できませんでした",
            key_technique="分析エラーのため取得できませんでした",
            validation_method="分析エラーのため取得できませんでした",
            experimental_results="分析エラーのため取得できませんでした",
            discussion_points=f"エラー: {error_msg}",
            implementation_details="分析エラーのため取得できませんでした",
            why_selected=f"検索クエリ「{query}」に関連",
            applicability="分析エラーのため取得できませんでした",
            next_papers=[],
            evidence_map={}
        )
    
    def extract_sections(self, full_text: str) -> Dict[str, str]:
        """
        Extract sections from full text
        
        Args:
            full_text: Complete paper text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        
        # Common section patterns
        section_patterns = [
            r'(?i)^abstract[:\s]*',
            r'(?i)^introduction[:\s]*',
            r'(?i)^background[:\s]*',
            r'(?i)^related\s+work[:\s]*',
            r'(?i)^method(?:s|ology)?[:\s]*',
            r'(?i)^approach[:\s]*',
            r'(?i)^experiment(?:s|al)?[:\s]*',
            r'(?i)^result(?:s)?[:\s]*',
            r'(?i)^discussion[:\s]*',
            r'(?i)^conclusion[:\s]*',
            r'(?i)^future\s+work[:\s]*',
            r'(?i)^acknowledge?ment(?:s)?[:\s]*',
            r'(?i)^reference(?:s)?[:\s]*',
        ]
        
        import re
        
        # Split text by sections
        lines = full_text.split('\n')
        current_section = 'abstract'  # Default
        current_content = []
        
        for line in lines:
            # Check if line is a section header
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = line.strip().lower().rstrip(':').replace(' ', '_')
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # If no sections found, return full text as single section
        if not sections or len(sections) == 1:
            sections = {
                'full_text': full_text[:10000],  # Limit for safety
                'abstract': full_text[:1000]
            }
        
        return sections
    
    def determine_analysis_strategy(
        self, 
        total_tokens: int,
        importance_score: float = 5.0
    ) -> str:
        """
        Determine best analysis strategy based on paper characteristics
        
        Args:
            total_tokens: Total tokens in the paper
            importance_score: Relevance score of the paper
            
        Returns:
            Strategy: 'full', 'map_reduce', or 'abstract_only'
        """
        # Gemini's context limit (conservative estimate)
        GEMINI_CONTEXT_LIMIT = 30000  # Conservative for safety
        
        if total_tokens < 10000:
            # Short paper - analyze full text
            return 'full'
        elif total_tokens < GEMINI_CONTEXT_LIMIT and importance_score > 7.0:
            # Important paper that fits in context
            return 'full'
        elif total_tokens < 50000:
            # Medium paper - use map-reduce
            return 'map_reduce'
        elif importance_score > 8.0:
            # Very important but long paper - map-reduce with priority sections
            return 'map_reduce_priority'
        else:
            # Very long and not critical - abstract only
            return 'abstract_only'