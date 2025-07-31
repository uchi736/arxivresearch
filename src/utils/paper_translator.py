"""
Paper translation module with figure/table preservation

This module handles translation of academic papers while preserving
figures and tables by embedding page images.
"""

import os
import fitz  # PyMuPDF
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential


class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_minute: int = 15):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.api_calls = 0
        self.start_time = time.time()
    
    def wait(self):
        """Wait if necessary to respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"  - Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_call = time.time()
    
    def log_call(self):
        """Log API call for monitoring"""
        self.api_calls += 1
        elapsed = time.time() - self.start_time
        calls_per_min = self.api_calls / (elapsed / 60) if elapsed > 0 else 0
        print(f"  - API calls: {self.api_calls}, Rate: {calls_per_min:.1f}/min")
        
        # Warning if rate is too high
        if calls_per_min > 50:
            print("⚠️ WARNING: High API call rate detected!")


class PaperTranslator:
    """Translator for academic papers with figure preservation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize translator with Gemini API"""
        load_dotenv()
        
        # Get model config
        try:
            from src.core.config import get_model_config
            self.config = get_model_config()
        except:
            # Fallback if config module not available
            self.config = None
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize rate limiter with config value or default
        rate_limit = self.config.pdf_translation_rate_limit if self.config else 15
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit)
    
    def download_pdf(self, url: str, output_path: str) -> bool:
        """Download PDF from URL"""
        try:
            print(f"Downloading PDF from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return False
    
    def find_pages_with_visuals(self, pdf_path: str) -> List[int]:
        """Find all pages containing figures or tables"""
        doc = fitz.open(pdf_path)
        visual_pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Check for visual elements
            has_visuals = (
                len(page.get_images()) > 0 or
                len(page.get_drawings()) > 0 or
                'figure' in text.lower() or
                'table' in text.lower()
            )
            
            if has_visuals:
                visual_pages.append(page_num)
        
        doc.close()
        print(f"Found {len(visual_pages)} pages with visual elements")
        return visual_pages
    
    def extract_page_as_base64(self, pdf_path: str, page_num: int, zoom: float = 1.0) -> str:
        """Extract a page as base64 encoded image with optimized resolution"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Get page dimensions
        rect = page.rect
        width, height = rect.width, rect.height
        
        # Determine optimal zoom based on page size
        # Target max dimension from config or default 1200px
        target_max_dim = self.config.pdf_image_max_dimension if self.config else 1200
        max_dim = max(width, height)
        if max_dim > target_max_dim:
            zoom = min(zoom, target_max_dim / max_dim)
        
        # Get page as optimized image
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image for further optimization
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Convert to JPEG for smaller size (if not text-heavy)
        buffer = io.BytesIO()
        quality = self.config.pdf_image_quality if self.config else 85
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        
        # Convert to base64
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        doc.close()
        return f"data:image/jpeg;base64,{base64_img}"
    
    def convert_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
        """Convert PDF page to PIL Image with optimized DPI"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Lower DPI for better performance (150 instead of 300)
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        doc.close()
        
        # Load and optimize image
        img = Image.open(io.BytesIO(img_data))
        
        # Resize if too large
        max_size = (1600, 2000)
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return img
    
    def translate_text_section(self, text: str) -> Optional[str]:
        """Translate a text section with retry logic"""
        # Apply retry decorator dynamically based on config
        retry_attempts = self.config.pdf_retry_attempts if self.config else 3
        retry_decorator = retry(
            stop=stop_after_attempt(retry_attempts), 
            wait=wait_exponential(multiplier=1, min=4, max=60)
        )
        return retry_decorator(self._translate_text_section_impl)(text)
    
    def _translate_text_section_impl(self, text: str) -> Optional[str]:
        """Internal implementation of translate_text_section"""
        if not text.strip():
            return ""
        
        # Limit text to avoid token limits
        max_length = self.config.pdf_max_text_length if self.config else 4000
        text_to_translate = text[:max_length]
        
        prompt = f"""あなたは学術論文の専門翻訳者です。
以下の英語テキストを日本語に翻訳してください。
学術的なトーンと専門用語を維持してください。
適切なHTMLタグ（h1, h2, h3, p, ul, li等）で出力してください。

英語テキスト:
{text_to_translate}

日本語翻訳（HTML形式）:"""
        
        try:
            self.rate_limiter.wait()
            response = self.model.generate_content(prompt)
            self.rate_limiter.log_call()
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Resource exhausted" in error_msg:
                print(f"Rate limit hit, waiting 60s...")
                time.sleep(60)
                raise
            elif "413" in error_msg or "Request too large" in error_msg:
                # Try with even smaller text
                if len(text_to_translate) > 2000:
                    print(f"Text too large, reducing to 2000 chars...")
                    return self.translate_text_section(text[:2000])
                raise
            else:
                print(f"Error translating text: {error_msg}")
                raise
    
    def translate_visual_page_text_only(self, page_image: Image.Image, page_num: int) -> Optional[str]:
        """Translate only the text parts of a visual page with retry logic"""
        # Apply retry decorator dynamically based on config
        retry_attempts = self.config.pdf_retry_attempts if self.config else 3
        retry_decorator = retry(
            stop=stop_after_attempt(retry_attempts), 
            wait=wait_exponential(multiplier=1, min=4, max=60)
        )
        return retry_decorator(self._translate_visual_page_impl)(page_image, page_num)
    
    def _translate_visual_page_impl(self, page_image: Image.Image, page_num: int) -> Optional[str]:
        """Internal implementation of translate_visual_page_text_only"""
        prompt = """あなたは学術論文の専門翻訳者です。
このページのテキスト部分のみを日本語に翻訳してください。

重要な指示:
1. テキスト部分のみを翻訳（図表の説明は不要）
2. 図表がある場所には以下のプレースホルダーを挿入:
   - 図の場合: <div class="figure-placeholder" data-type="figure">図X: [キャプションの翻訳]</div>
   - 表の場合: <div class="figure-placeholder" data-type="table">表X: [キャプションの翻訳]</div>
3. 本文は通常通りHTMLタグで構造化
4. 図表の内容説明は含めない

このページのテキストを翻訳してください:"""
        
        try:
            self.rate_limiter.wait()
            response = self.model.generate_content([prompt, page_image])
            self.rate_limiter.log_call()
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Resource exhausted" in error_msg:
                print(f"Rate limit hit on page {page_num+1}, waiting 60s...")
                time.sleep(60)
                raise
            elif "413" in error_msg or "Request too large" in error_msg:
                print(f"Image too large for page {page_num+1}, skipping visual translation...")
                # Return a placeholder indicating the page couldn't be processed
                return f"<p><em>[ページ {page_num+1}: 画像サイズが大きすぎるため、テキスト翻訳をスキップしました]</em></p>"
            else:
                print(f"Error translating page {page_num+1}: {error_msg}")
                raise
    
    def translate_paper(self, pdf_path: str, output_html: str) -> bool:
        """Translate entire paper with embedded figures"""
        print("Starting paper translation...")
        
        # Find pages with visuals
        visual_pages = self.find_pages_with_visuals(pdf_path)
        
        # Process paper
        doc = fitz.open(pdf_path)
        translated_sections = []
        
        # Group pages into sections
        current_text = ""
        current_start = 0
        
        for page_num in range(len(doc)):
            if page_num in visual_pages:
                # Save accumulated text
                if current_text.strip():
                    print(f"Translating text pages {current_start+1}-{page_num}...")
                    translation = self.translate_text_section(current_text)
                    if translation:
                        translated_sections.append({
                            'type': 'text',
                            'pages': f"{current_start+1}-{page_num}",
                            'content': translation
                        })
                    current_text = ""
                
                # Process visual page
                print(f"Processing visual page {page_num+1}...")
                page_image = self.convert_page_to_image(pdf_path, page_num)
                
                # Get the full page as embedded image
                page_image_base64 = self.extract_page_as_base64(pdf_path, page_num)
                
                # Translate text and identify figures
                translation = self.translate_visual_page_text_only(page_image, page_num)
                
                if translation:
                    translated_sections.append({
                        'type': 'visual',
                        'page': page_num + 1,
                        'content': translation,
                        'page_image': page_image_base64
                    })
                
                current_start = page_num + 1
            else:
                # Accumulate text
                page = doc[page_num]
                current_text += page.get_text() + "\n"
        
        # Don't forget last section
        if current_text.strip():
            print(f"Translating text pages {current_start+1}-{len(doc)}...")
            translation = self.translate_text_section(current_text)
            if translation:
                translated_sections.append({
                    'type': 'text',
                    'pages': f"{current_start+1}-{len(doc)}",
                    'content': translation
                })
        
        doc.close()
        
        # Create HTML output
        self._create_html_output(translated_sections, output_html)
        
        print(f"Translation complete! Output saved to {output_html}")
        return True
    
    def _create_html_output(self, sections: List[Dict[str, Any]], output_path: str):
        """Create HTML output with embedded figures"""
        html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>翻訳論文</title>
    <style>
        body {{
            font-family: "Hiragino Kaku Gothic Pro", "ヒラギノ角ゴ Pro", "Yu Gothic", "游ゴシック", "Meiryo", "メイリオ", sans-serif;
            line-height: 1.8;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 10px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.2em;
        }}
        h2 {{
            color: #0066cc;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 5px solid #0066cc;
            padding-left: 15px;
        }}
        h3 {{
            color: #333;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        .page-image {{
            width: 100%;
            margin: 30px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .figure-placeholder {{
            background-color: #e3f2fd;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            color: #1565c0;
        }}
        .page-info {{
            background-color: #e8f5e9;
            padding: 10px;
            margin: 30px 0;
            border-radius: 5px;
            font-size: 0.9em;
            text-align: center;
            color: #2e7d32;
        }}
        .visual-page {{
            margin: 40px 0;
            padding: 30px;
            background-color: #fafafa;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }}
        .page-content {{
            margin-bottom: 30px;
        }}
        .translation-note {{
            background-color: #fff9c4;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            font-size: 0.95em;
            color: #666;
            border-left: 4px solid #fbc02d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="translation-note">
            <p><strong>翻訳情報</strong></p>
            <p>このドキュメントは、Gemini 2.0 Flash Vision APIを使用して翻訳されました。</p>
            <p>図表を含むページは元の画像をそのまま表示しています。</p>
            <p>翻訳日時: {timestamp}</p>
        </div>
        
        {content}
    </div>
</body>
</html>"""
        
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        content_html = ""
        for section in sections:
            if section['type'] == 'text':
                content_html += f'<div class="page-info">ページ {section["pages"]}</div>\n'
                content_html += f'<div class="page-content">{section["content"]}</div>\n'
            else:  # visual
                content_html += f'<div class="visual-page">\n'
                content_html += f'<div class="page-info">ページ {section["page"]}（図表を含む）</div>\n'
                content_html += f'<div class="page-content">{section["content"]}</div>\n'
                content_html += f'<img src="{section["page_image"]}" alt="Page {section["page"]}" class="page-image">\n'
                content_html += f'</div>\n'
        
        final_html = html_template.format(
            content=content_html,
            timestamp=timestamp
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
    
    def translate_from_url(self, pdf_url: str, output_html: str) -> bool:
        """Download and translate paper from URL"""
        # Create temp file for PDF
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Download PDF
            if not self.download_pdf(pdf_url, tmp_path):
                return False
            
            # Translate
            result = self.translate_paper(tmp_path, output_html)
            
            # Cleanup
            os.unlink(tmp_path)
            
            return result
        except Exception as e:
            print(f"Error during translation: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False


# Convenience function for backward compatibility
def translate_paper(pdf_path_or_url: str, output_html: str, api_key: Optional[str] = None) -> bool:
    """
    Translate a paper from file path or URL
    
    Args:
        pdf_path_or_url: Local PDF path or URL
        output_html: Output HTML file path
        api_key: Optional Gemini API key
        
    Returns:
        True if successful, False otherwise
    """
    translator = PaperTranslator(api_key)
    
    if pdf_path_or_url.startswith('http'):
        return translator.translate_from_url(pdf_path_or_url, output_html)
    else:
        return translator.translate_paper(pdf_path_or_url, output_html)