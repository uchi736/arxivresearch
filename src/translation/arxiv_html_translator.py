"""
ArXiv HTML Translator using ar5iv service
Efficient translation by preserving LaTeX structure and using batch processing
"""
import os
import re
import time
import json
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString, Tag

from src.core.config import config, create_llm_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivHTMLTranslator:
    """
    ar5ivのHTML版論文を効率的に翻訳
    - HTML構造を保持
    - 数式・図表参照を維持
    - バッチ処理で高速化
    """
    
    def __init__(self):
        """Initialize with Vertex AI model"""
        self.model = create_llm_model()
        self.batch_size = 15  # 一度に翻訳する要素数
        logger.info("ArxivHTMLTranslator initialized")
        
    def arxiv_to_ar5iv_url(self, arxiv_url: str) -> str:
        """
        Convert arXiv URL to ar5iv HTML URL
        
        Examples:
            https://arxiv.org/abs/2312.12345 -> https://ar5iv.labs.arxiv.org/html/2312.12345
            https://arxiv.org/pdf/2312.12345.pdf -> https://ar5iv.labs.arxiv.org/html/2312.12345
        """
        # Extract arXiv ID
        arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
        if not arxiv_id_match:
            raise ValueError(f"Invalid arXiv URL: {arxiv_url}")
            
        arxiv_id = arxiv_id_match.group(1)
        return f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        
    def fetch_ar5iv_html(self, arxiv_url: str) -> str:
        """
        Fetch HTML from ar5iv
        
        Args:
            arxiv_url: Original arXiv URL
            
        Returns:
            HTML content
        """
        ar5iv_url = self.arxiv_to_ar5iv_url(arxiv_url)
        logger.info(f"Fetching HTML from {ar5iv_url}")
        
        response = requests.get(ar5iv_url, timeout=30)
        response.raise_for_status()
        
        # Ensure proper encoding
        response.encoding = 'utf-8'
        
        return response.text
        
    def protect_special_elements(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Protect math, citations, and references from translation
        
        Args:
            text: Original text
            
        Returns:
            (protected_text, placeholder_map)
        """
        placeholders = {}
        protected = text
        
        # Protect inline math $...$
        math_pattern = r'\$[^$]+\$'
        for i, match in enumerate(re.finditer(math_pattern, protected)):
            placeholder = f"__MATH_{i}__"
            placeholders[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)
            
        # Protect citations [1], [2,3], etc.
        citation_pattern = r'\[\d+(?:,\s*\d+)*\]'
        for i, match in enumerate(re.finditer(citation_pattern, protected)):
            placeholder = f"__CITE_{i}__"
            placeholders[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)
            
        # Protect figure/table references
        ref_pattern = r'(Figure|Table|Section|Equation|Fig\.|Tab\.|Sec\.|Eq\.)\s*\d+(?:\.\d+)*'
        for i, match in enumerate(re.finditer(ref_pattern, protected, re.IGNORECASE)):
            placeholder = f"__REF_{i}__"
            placeholders[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)
            
        return protected, placeholders
        
    def restore_special_elements(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore protected elements"""
        restored = text
        for placeholder, original in placeholders.items():
            restored = restored.replace(placeholder, original)
        return restored
        
    def extract_translatable_elements(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract elements that need translation
        
        Returns:
            List of dictionaries with element info
        """
        elements = []
        
        # Title
        title_elem = soup.find('h1', class_='ltx_title')
        if title_elem and title_elem.text.strip():
            elements.append({
                'type': 'title',
                'text': title_elem.text.strip(),
                'element': title_elem
            })
            
        # Abstract
        abstract = soup.find('div', class_='ltx_abstract')
        if abstract:
            abstract_text = abstract.find('p')
            if abstract_text and abstract_text.text.strip():
                elements.append({
                    'type': 'abstract',
                    'text': abstract_text.text.strip(),
                    'element': abstract_text
                })
                
        # Section headers
        for header in soup.find_all(['h2', 'h3', 'h4'], class_=re.compile('ltx_title')):
            if header.text.strip():
                # Skip if it's just a number
                text = header.text.strip()
                if not re.match(r'^\d+\.?\d*$', text):
                    elements.append({
                        'type': 'header',
                        'text': text,
                        'element': header
                    })
                    
        # Paragraphs
        for para in soup.find_all('p', class_='ltx_p'):
            text = para.text.strip()
            if text and len(text) > 20:  # Skip very short paragraphs
                elements.append({
                    'type': 'paragraph',
                    'text': text,
                    'element': para
                })
                
        # Figure/Table captions
        for caption in soup.find_all(['figcaption', 'caption']):
            if caption.text.strip():
                elements.append({
                    'type': 'caption',
                    'text': caption.text.strip(),
                    'element': caption
                })
                
        logger.info(f"Extracted {len(elements)} translatable elements")
        return elements
        
    def translate_batch(self, elements: List[Dict]) -> List[str]:
        """
        Translate multiple elements in one API call
        
        Args:
            elements: List of element dictionaries
            
        Returns:
            List of translated texts
        """
        if not elements:
            return []
            
        # Prepare batch with protected elements
        batch_data = []
        for elem in elements:
            protected_text, placeholders = self.protect_special_elements(elem['text'])
            batch_data.append({
                'id': len(batch_data),
                'type': elem['type'],
                'text': protected_text,
                'placeholders': placeholders
            })
            
        # Create prompt
        prompt = f"""以下の学術論文の要素を日本語に翻訳してください。

【重要な指示】
1. 各要素のIDと種類を保持してJSON形式で返答
2. __MATH_X__, __CITE_X__, __REF_X__ などのプレースホルダーは変更しない
3. 学術的で正確な翻訳を心がける
4. 専門用語は適切な日本語訳を使用

【入力データ】
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

【出力形式】
{{
  "translations": [
    {{"id": 0, "text": "翻訳されたテキスト"}},
    {{"id": 1, "text": "翻訳されたテキスト"}},
    ...
  ]
}}
"""
        
        try:
            response = self.model.invoke(prompt)
            content = response.content
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
                
            result = json.loads(content)
            
            # Sort by ID and extract translations
            translations = sorted(result['translations'], key=lambda x: x['id'])
            
            # Restore placeholders
            final_translations = []
            for i, trans in enumerate(translations):
                if i < len(batch_data):
                    restored = self.restore_special_elements(
                        trans['text'], 
                        batch_data[i]['placeholders']
                    )
                    final_translations.append(restored)
                    
            return final_translations
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {response.content[:500]}...")
            # Fallback: return original texts
            return [elem['text'] for elem in elements]
        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            # Fallback: return original texts
            return [elem['text'] for elem in elements]
            
    def update_html_with_translations(self, soup: BeautifulSoup, elements: List[Dict], translations: List[str]):
        """
        Update HTML elements with translated text
        """
        for elem_info, translation in zip(elements, translations):
            element = elem_info['element']
            
            # Clear existing content
            element.clear()
            
            # Add translated text - ensure it's properly handled as unicode
            # NavigableString is already imported at the top
            element.append(NavigableString(translation))
            
    def translate_arxiv_paper(self, arxiv_url: str, output_path: str) -> bool:
        """
        Main translation function
        
        Args:
            arxiv_url: arXiv paper URL
            output_path: Path to save translated HTML
            
        Returns:
            Success status
        """
        logger.info(f"Starting translation of {arxiv_url}")
        start_time = time.time()
        
        try:
            # Fetch HTML
            html_content = self.fetch_ar5iv_html(arxiv_url)
            # Use lxml parser for better unicode handling if available, fallback to html.parser
            try:
                soup = BeautifulSoup(html_content, 'lxml')
            except:
                soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract translatable elements
            elements = self.extract_translatable_elements(soup)
            
            # Process in batches
            total_batches = (len(elements) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(elements), self.batch_size):
                batch = elements[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Translate batch
                translations = self.translate_batch(batch)
                
                # Update HTML
                self.update_html_with_translations(soup, batch, translations)
                
                # Brief pause to avoid rate limits
                if i + self.batch_size < len(elements):
                    time.sleep(1)
                    
            # Add translation metadata
            meta_tag = soup.new_tag('meta', attrs={
                'name': 'translation-info',
                'content': f'Translated by ArxivHTMLTranslator on {datetime.now().isoformat()}'
            })
            soup.head.append(meta_tag)
            
            # Ensure UTF-8 encoding in meta tag
            if soup.head:
                # Remove existing charset meta tags
                for meta in soup.head.find_all('meta', attrs={'charset': True}):
                    meta.decompose()
                for meta in soup.head.find_all('meta', attrs={'http-equiv': 'content-type'}):
                    meta.decompose()
                    
                # Add UTF-8 meta tag at the beginning
                charset_meta = soup.new_tag('meta', charset='utf-8')
                soup.head.insert(0, charset_meta)
            
            # Save translated HTML
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                # Use prettify() for better formatting and encoding
                f.write(soup.prettify())
                
            elapsed = time.time() - start_time
            logger.info(f"Translation completed in {elapsed:.1f}s")
            logger.info(f"Saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return False
            
    def check_ar5iv_availability(self, arxiv_url: str) -> bool:
        """Check if paper is available on ar5iv"""
        try:
            ar5iv_url = self.arxiv_to_ar5iv_url(arxiv_url)
            response = requests.head(ar5iv_url, timeout=10)
            return response.status_code == 200
        except:
            return False