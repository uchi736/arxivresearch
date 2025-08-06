"""
Structured Ochiai analyzer using Pydantic for reliable JSON output
"""

import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from src.core.models import OchiaiFormatAdvanced, PaperMetadata
from src.core.config import create_llm_model


class OchiaiStructuredAnalyzer:
    """Analyzer that ensures structured output using Pydantic"""
    
    def __init__(self, model=None):
        """Initialize with LLM model"""
        self.model = model or create_llm_model()
    
    def analyze_with_validation(
        self,
        text: str,
        paper_metadata: PaperMetadata,
        query: str,
        max_retries: int = 3
    ) -> OchiaiFormatAdvanced:
        """
        Analyze text and ensure valid Ochiai format output
        
        Args:
            text: Paper text to analyze
            paper_metadata: Paper metadata
            query: Original search query
            max_retries: Maximum retry attempts
            
        Returns:
            OchiaiFormatAdvanced object
        """
        
        # Create structured prompt
        prompt = self._create_structured_prompt(text, paper_metadata, query)
        
        # Try to get valid response
        for attempt in range(max_retries):
            try:
                # Get LLM response
                response = self.model.invoke(prompt)
                content = response.content.strip()
                
                # Parse JSON
                json_data = self._extract_json(content)
                
                # Validate and create Ochiai object
                ochiai = self._create_ochiai_from_json(json_data, paper_metadata, query)
                
                return ochiai
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # Add more explicit instructions
                    prompt = self._add_json_clarification(prompt, str(e))
                else:
                    # Final attempt failed - return fallback
                    return self._create_fallback_ochiai(paper_metadata, query, f"JSON parse error: {e}")
                    
            except ValidationError as e:
                if attempt < max_retries - 1:
                    # Add field requirements
                    prompt = self._add_field_requirements(prompt, e)
                else:
                    # Final attempt failed - return fallback
                    return self._create_fallback_ochiai(paper_metadata, query, f"Validation error: {e}")
                    
            except Exception as e:
                # Unexpected error - return fallback
                return self._create_fallback_ochiai(paper_metadata, query, str(e))
        
        # Should not reach here
        return self._create_fallback_ochiai(paper_metadata, query, "Max retries exceeded")
    
    def _create_structured_prompt(self, text: str, paper_metadata: PaperMetadata, query: str) -> str:
        """Create a structured prompt for Ochiai analysis"""
        
        # Define the exact JSON structure with detailed examples
        json_template = {
            "what_is_it": "この研究は[具体的な研究対象とアプローチ]を提案し、[解決しようとする具体的な問題]に取り組んでいる。[研究の背景と動機]",
            "comparison_with_prior_work": "従来手法[具体的な手法名1,2,3]と比較して、本研究は[1.具体的な新規性] [2.技術的な差別化点] [3.性能面での優位性]を持つ。特に[最も重要な差別化要素]が革新的である",
            "key_technique": "提案手法の核心技術は[1.主要技術名とその詳細説明] [2.アルゴリズムの具体的な処理フロー] [3.数式やパラメータの説明]である。これにより[期待される効果と理論的根拠]を実現",
            "validation_method": "実験設定：[使用データセット名と規模]、[評価指標1,2,3の詳細]、[ベースライン手法の詳細]、[実験環境：GPU/CPU、フレームワーク]、[ハイパーパラメータ設定]",
            "experimental_results": "定量的結果：[指標1: 具体的数値と改善率]、[指標2: 具体的数値と改善率]、[統計的有意性]。定性的結果：[視覚的な改善例]、[エラー分析]、[成功/失敗ケースの詳細分析]",
            "discussion_points": "考察：[1.なぜ提案手法が有効だったかの理論的説明] [2.失敗ケースの原因分析] [3.計算コストとのトレードオフ]。限界：[1.適用範囲の制約] [2.スケーラビリティの課題]",
            "implementation_details": "実装詳細：[使用言語/フレームワーク]、[コードの可用性とURL]、[主要な実装上の工夫点]、[計算複雑度：時間O(?)、空間O(?)]、[並列化/最適化手法]",
            "why_selected": f"検索クエリ「{query}」との関連性：[1.直接的な技術的関連] [2.応用可能性] [3.理論的な貢献]の観点から高い関連性を持つ",
            "applicability": "応用可能性：[1.具体的な産業応用例：業界名と用途] [2.他の研究分野への転用可能性] [3.実用化に向けた課題と解決策]",
            "next_papers": ["[具体的な関連論文1: タイトルと簡単な説明]", "[具体的な関連論文2: タイトルと簡単な説明]", "[具体的な関連論文3: タイトルと簡単な説明]"]
        }
        
        prompt = f"""以下の論文を落合陽一フォーマットで分析してください。

論文タイトル: {paper_metadata.title}
著者: {', '.join(paper_metadata.authors[:3])}

分析対象テキスト:
{text[:15000]}...

以下のJSON形式で回答してください。各フィールドは具体的で詳細な日本語で記述してください：

```json
{json.dumps(json_template, ensure_ascii=False, indent=2)}
```

重要な指示:
1. 必ず上記のJSON形式を厳密に守ってください
2. 各フィールドは具体的で詳細な内容を含む完全な文章にしてください（最低100文字以上）
3. [括弧内]の部分を実際の具体的な内容に置き換えてください - 抽象的な記述は避ける
4. 数値、手法名、データセット名などは必ず具体的に記載してください
5. experimental_resultsには必ず定量的な数値結果を含めてください（%、スコア、時間など）
6. discussion_pointsには「なぜ」の観点から深い考察を含めてください
7. next_papersは実際に存在する関連論文タイトルを3つ以上挙げてください
8. JSONの前後に余分なテキストを含めないでください

分析を開始してください："""
        
        return prompt
    
    def _extract_json(self, content: str) -> Dict:
        """Extract JSON from LLM response"""
        # Remove markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        
        # Parse JSON
        return json.loads(content.strip())
    
    def _create_ochiai_from_json(
        self, 
        json_data: Dict, 
        paper_metadata: PaperMetadata,
        query: str
    ) -> OchiaiFormatAdvanced:
        """Create OchiaiFormatAdvanced from JSON data with validation"""
        
        # Ensure all required fields exist
        required_fields = [
            "what_is_it", "comparison_with_prior_work", "key_technique",
            "validation_method", "experimental_results", "discussion_points",
            "implementation_details", "why_selected", "applicability", "next_papers"
        ]
        
        for field in required_fields:
            if field not in json_data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Create Ochiai object
        return OchiaiFormatAdvanced(
            what_is_it=str(json_data["what_is_it"]),
            comparison_with_prior_work=str(json_data["comparison_with_prior_work"]),
            key_technique=str(json_data["key_technique"]),
            validation_method=str(json_data["validation_method"]),
            experimental_results=str(json_data["experimental_results"]),
            discussion_points=str(json_data["discussion_points"]),
            implementation_details=str(json_data["implementation_details"]),
            why_selected=str(json_data["why_selected"]),
            applicability=str(json_data["applicability"]),
            next_papers=list(json_data["next_papers"]) if isinstance(json_data["next_papers"], list) else [],
            evidence_map={}  # Not used in simple version
        )
    
    def _add_json_clarification(self, prompt: str, error: str) -> str:
        """Add clarification for JSON errors"""
        return prompt + f"""

JSONパースエラーが発生しました: {error}

以下の点を確認してください：
- 必ず```json と ``` で囲まれた有効なJSON形式で回答
- すべての文字列は二重引用符で囲む
- カンマの位置が正しいか確認
- 最後の要素の後にカンマを付けない"""
    
    def _add_field_requirements(self, prompt: str, error: ValidationError) -> str:
        """Add field requirements for validation errors"""
        return prompt + f"""

フィールド検証エラー: {error}

すべての必須フィールドを含めて、完全なJSON形式で回答してください。"""
    
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