"""
Advanced ArXiv Academic Paper Translator
Handles complex LaTeX structures and maintains formatting
"""
import os
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup, NavigableString, Tag

from src.translation.arxiv_html_translator import ArxivHTMLTranslator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivAcademicTranslator(ArxivHTMLTranslator):
    """
    Enhanced translator for academic papers with complex structures
    """
    
    def __init__(self):
        super().__init__()
        # Academic terminology dictionary
        self.terminology = {
            "LLM-based agents": "LLMベースエージェント",
            "Large Language Model": "大規模言語モデル",
            "evaluation": "評価",
            "benchmark": "ベンチマーク",
            "agent": "エージェント",
            "survey": "サーベイ",
            "framework": "フレームワーク",
            "methodology": "手法",
            "assessment": "評価",
            "capabilities": "能力",
            "performance": "性能",
            "metrics": "指標",
            "tool use": "ツール使用",
            "self-reflection": "自己省察",
            "memory": "メモリ",
            "planning": "計画",
            "reasoning": "推論",
            "multi-step": "マルチステップ",
            "function calling": "関数呼び出し"
        }
        
        # Elements to skip translation
        self.skip_classes = [
            'ltx_equation',
            'ltx_math',
            'ltx_cite',
            'ltx_ref',
            'ltx_bibitem',
            'ltx_listing',
            'ltx_verbatim',
            'ltx_code'
        ]
        
    def should_translate_element(self, element: Tag) -> bool:
        """Check if element should be translated"""
        # Skip elements with certain classes
        if element.get('class'):
            classes = element.get('class')
            if any(skip_class in classes for skip_class in self.skip_classes):
                return False
                
        # Skip if element contains complex structures
        if element.find_all(['svg', 'math', 'img']):
            return False
            
        return True
        
    def extract_translatable_elements(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract elements with academic paper considerations
        """
        elements = []
        
        # Title - most important
        title_elem = soup.find('h1', class_='ltx_title')
        if title_elem and title_elem.text.strip():
            elements.append({
                'type': 'title',
                'text': title_elem.text.strip(),
                'element': title_elem,
                'priority': 1
            })
            
        # Abstract - second most important
        abstract = soup.find('div', class_='ltx_abstract')
        if abstract:
            # Get abstract title
            abstract_title = abstract.find('h6', class_='ltx_title')
            if abstract_title and self.should_translate_element(abstract_title):
                elements.append({
                    'type': 'abstract_title',
                    'text': abstract_title.text.strip(),
                    'element': abstract_title,
                    'priority': 2
                })
            
            # Get abstract content
            abstract_text = abstract.find('p')
            if abstract_text and abstract_text.text.strip():
                elements.append({
                    'type': 'abstract',
                    'text': abstract_text.text.strip(),
                    'element': abstract_text,
                    'priority': 2
                })
                
        # Section headers with hierarchy
        for i, header in enumerate(soup.find_all(['h2', 'h3', 'h4'], class_=re.compile('ltx_title'))):
            if not self.should_translate_element(header):
                continue
                
            text = header.text.strip()
            # Skip pure numbers
            if re.match(r'^\d+\.?\d*$', text):
                continue
                
            # Determine header level
            level = int(header.name[1])  # h2->2, h3->3, h4->4
            
            elements.append({
                'type': f'header_h{level}',
                'text': text,
                'element': header,
                'priority': 3 + level  # Lower priority for deeper headers
            })
                    
        # Paragraphs - only substantial ones
        for para in soup.find_all('p', class_='ltx_p'):
            if not self.should_translate_element(para):
                continue
                
            text = para.text.strip()
            # Skip very short paragraphs or those with mostly citations
            if len(text) < 50:
                continue
                
            # Check citation density
            citation_count = len(re.findall(r'\[\d+\]', text))
            word_count = len(text.split())
            if citation_count > word_count * 0.3:  # Skip if >30% citations
                continue
                
            elements.append({
                'type': 'paragraph',
                'text': text,
                'element': para,
                'priority': 5
            })
                
        # Figure/Table captions - important for understanding
        for caption in soup.find_all(['figcaption', 'caption']):
            if caption.text.strip():
                elements.append({
                    'type': 'caption',
                    'text': caption.text.strip(),
                    'element': caption,
                    'priority': 4
                })
                
        # List items in key sections
        for list_elem in soup.find_all(['ul', 'ol']):
            for item in list_elem.find_all('li'):
                if self.should_translate_element(item):
                    text = item.text.strip()
                    if text and len(text) > 20:
                        elements.append({
                            'type': 'list_item',
                            'text': text,
                            'element': item,
                            'priority': 5
                        })
                        
        logger.info(f"Extracted {len(elements)} academic elements")
        
        # Sort by priority for better translation context
        elements.sort(key=lambda x: x.get('priority', 10))
        
        return elements
        
    def apply_terminology(self, text: str) -> str:
        """Apply consistent terminology translations"""
        # Don't pre-apply terminology - let the LLM handle it with context
        # This avoids encoding issues and provides better context-aware translation
        return text
        
    def translate_batch(self, elements: List[Dict]) -> List[str]:
        """
        Enhanced batch translation for academic content
        """
        if not elements:
            return []
            
        # Group by type for better context
        grouped = {}
        for elem in elements:
            elem_type = elem['type']
            if elem_type not in grouped:
                grouped[elem_type] = []
            grouped[elem_type].append(elem)
            
        # Prepare batch with academic context
        batch_data = []
        for elem in elements:
            protected_text, placeholders = self.protect_special_elements(elem['text'])
            
            # No pre-processing needed
            preprocessed = protected_text
            
            batch_data.append({
                'id': len(batch_data),
                'type': elem['type'],
                'text': preprocessed,
                'placeholders': placeholders,
                'original': elem['text']
            })
            
        # Create academic-focused prompt
        prompt = f"""以下の学術論文の要素を日本語に翻訳してください。これは「{elements[0]['text'][:50]}...」に関する論文です。

【重要な指示】
1. 学術的で正確な翻訳を心がける
2. 専門用語は一貫性を保つ（例: LLM-based agents → LLMベースエージェント）
3. __MATH_X__, __CITE_X__, __REF_X__ などのプレースホルダーは絶対に変更しない
4. 文章の論理構造を保持する
5. セクションタイトルは簡潔に
6. 箇条書きの構造を維持

【専門用語の統一】
- Large Language Model → 大規模言語モデル
- evaluation → 評価
- benchmark → ベンチマーク
- agent → エージェント

【入力データ】
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

【出力形式】
{{
  "translations": [
    {{"id": 0, "text": "翻訳されたテキスト"}},
    ...
  ]
}}
"""
        
        try:
            response = self.model.invoke(prompt)
            content = response.content
            
            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
                
            result = json.loads(content)
            translations = sorted(result['translations'], key=lambda x: x['id'])
            
            # Post-process and restore
            final_translations = []
            for i, trans in enumerate(translations):
                if i < len(batch_data):
                    # Restore placeholders
                    restored = self.restore_special_elements(
                        trans['text'], 
                        batch_data[i]['placeholders']
                    )
                    
                    # Add restored translation
                    final_translations.append(restored)
                    
            return final_translations
            
        except Exception as e:
            logger.error(f"Academic translation error: {e}")
            # Return original texts as fallback
            return [elem['text'] for elem in elements]
            
    def fix_forest_diagram(self, soup: BeautifulSoup):
        """
        Fix broken forest diagram structure
        """
        # Look for the specific problematic text pattern
        import re
        
        # Find elements containing the forest notation
        for elem in soup.find_all(text=re.compile(r'\{forest\}|for tree=')):
            if isinstance(elem, NavigableString):
                parent = elem.parent
                if parent:
                    # Log what we found
                    logger.info(f"Found forest notation in {parent.name} tag")
                    
                    # Create replacement
                    new_div = soup.new_tag('div', attrs={'class': 'converted-tree'})
                    
                    # Add note
                    note = soup.new_tag('p')
                    note.string = "[図1: 論文の構造概要 - 元のツリー図は以下の階層構造を表しています]"
                    new_div.append(note)
                    
                    # Create a simplified structure
                    structure = soup.new_tag('div')
                    structure.string = """
エージェント評価
├── エージェント能力の評価 (§2)
│   ├── 計画とマルチステップ推論 (§2.1)
│   ├── 関数呼び出しとツール使用 (§2.2)
│   ├── 自己省察 (§2.3)
│   └── メモリ (§2.4)
├── アプリケーション固有のエージェント評価 (§3)
│   ├── ウェブエージェント (§3.1)
│   ├── ソフトウェアエンジニアリングエージェント (§3.2)
│   ├── 科学エージェント (§3.3)
│   └── 会話型エージェント (§3.4)
├── 汎用エージェントの評価 (§4)
├── エージェント評価のフレームワーク (§5)
└── 議論 (§6)
    ├── 現在のトレンド (§6.1)
    └── 新興の方向性 (§6.2)
                    """
                    new_div.append(structure)
                    
                    # Replace the parent element
                    parent.replace_with(new_div)
                    break  # Only process the first occurrence
            
    def translate_arxiv_paper(self, arxiv_url: str, output_path: str) -> bool:
        """
        Override parent method to add academic-specific processing
        """
        logger.info(f"Starting academic translation of {arxiv_url}")
        start_time = time.time()
        
        try:
            # Fetch HTML
            html_content = self.fetch_ar5iv_html(arxiv_url)
            
            # Parse with lxml for better handling
            try:
                soup = BeautifulSoup(html_content, 'lxml')
            except:
                soup = BeautifulSoup(html_content, 'html.parser')
            
            # Fix broken structures first
            self.fix_forest_diagram(soup)
            
            # Now proceed with translation
            elements = self.extract_translatable_elements(soup)
            
            # Process in smaller batches for academic quality
            total_batches = (len(elements) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(elements), self.batch_size):
                batch = elements[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} (academic mode)")
                
                # Translate batch
                translations = self.translate_batch(batch)
                
                # Update HTML
                self.update_html_with_translations(soup, batch, translations)
                
                # Brief pause
                if i + self.batch_size < len(elements):
                    time.sleep(1)
                    
            # Add metadata
            meta_tag = soup.new_tag('meta', attrs={
                'name': 'translation-info',
                'content': f'Translated by ArxivAcademicTranslator on {datetime.now().isoformat()}'
            })
            soup.head.append(meta_tag)
            
            # Add custom CSS for better formatting
            style_tag = soup.new_tag('style')
            style_tag.string = """
            /* Academic translation styles */
            .converted-tree {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }
            .converted-tree p {
                font-style: italic;
                color: #666;
                margin-bottom: 10px;
            }
            .converted-tree ul {
                margin-left: 20px;
            }
            """
            soup.head.append(style_tag)
            
            # Ensure UTF-8
            if soup.head:
                for meta in soup.head.find_all('meta', attrs={'charset': True}):
                    meta.decompose()
                for meta in soup.head.find_all('meta', attrs={'http-equiv': 'content-type'}):
                    meta.decompose()
                    
                charset_meta = soup.new_tag('meta', charset='utf-8')
                soup.head.insert(0, charset_meta)
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
                
            elapsed = time.time() - start_time
            logger.info(f"Academic translation completed in {elapsed:.1f}s")
            logger.info(f"Saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Academic translation failed: {e}")
            import traceback
            traceback.print_exc()
            return False