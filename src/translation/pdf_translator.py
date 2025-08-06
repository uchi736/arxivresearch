"""
PDF Translation Module using Vertex AI
"""
import os
import io
import time
import requests
from datetime import datetime
from typing import Optional, List, Dict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[WARNING] PyMuPDF not installed. PDF translation features will be unavailable.")
    fitz = None

from PIL import Image

from src.core.config import config, create_llm_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFTranslatorWithReportLab:
    """
    PDF翻訳クラス
    PyMuPDFを使用してPDFからテキストと画像を抽出し、
    Gemini Visionモデルで日本語に翻訳します。
    """
    
    def __init__(self):
        """
        Initialize translator with Vertex AI
        """
        if not fitz:
            raise ImportError("PyMuPDF is required for PDF translation. Install with: pip install PyMuPDF")
            
        # Use Vertex AI through LangChain
        self.model = create_llm_model()
        logger.info("PDFTranslatorWithReportLab initialized with Vertex AI")
        
    def download_pdf(self, url: str, output_path: Optional[str] = None) -> str:
        """
        PDFをダウンロード
        
        Args:
            url: PDFのURL
            output_path: 保存先パス（省略時は一時ディレクトリ）
            
        Returns:
            保存されたファイルのパス
        """
        if not output_path:
            os.makedirs("temp", exist_ok=True)
            filename = url.split('/')[-1]
            output_path = f"temp/{filename}"
            
        logger.info(f"Downloading PDF from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"PDF downloaded to {output_path}")
        return output_path
        
    def extract_page_content(self, page) -> Dict:
        """
        ページからテキストと画像を抽出
        
        Args:
            page: PyMuPDFのページオブジェクト
            
        Returns:
            {"text": str, "images": List[PIL.Image]}
        """
        # テキスト抽出
        text = page.get_text()
        
        # 画像抽出
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # 画像データを取得
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # PILイメージに変換
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                pil_image = Image.open(io.BytesIO(img_data))
                
                # サイズ最適化（大きすぎる場合はリサイズ）
                max_size = (1024, 1024)
                if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
                    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                images.append(pil_image)
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index}: {e}")
                
        return {"text": text, "images": images}
        
    def translate_page_content(self, content: Dict, page_num: int) -> str:
        """
        ページコンテンツを翻訳
        
        Args:
            content: extract_page_contentの出力
            page_num: ページ番号
            
        Returns:
            翻訳されたHTML
        """
        # テキストの前処理（長すぎる場合は切り詰める）
        text_content = content['text'][:3000] if content['text'] else ""
        
        prompt = f"""
以下は学術論文のページ{page_num}の内容です。
テキストと画像が含まれている場合があります。

【タスク】
1. テキスト部分を自然な日本語に翻訳してください
2. 図表のキャプションも翻訳してください
3. 数式や化学式などは元のまま残してください
4. 翻訳結果をHTML形式で出力してください
5. 画像への参照は[図{page_num}-X]の形式で記載してください

【テキスト】
{text_content}

【出力形式】
<div class="page">
  <h3>ページ {page_num}</h3>
  <!-- 翻訳されたコンテンツ -->
</div>
"""
        
        try:
            # LangChainモデルはテキストのみをサポート（画像は将来的に対応）
            if content['images']:
                logger.info(f"Page {page_num} contains {len(content['images'])} images (text-only translation)")
            
            # LangChainモデルで翻訳
            response = self.model.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Translation error on page {page_num}: {e}")
            return f'<div class="page error">ページ {page_num} の翻訳に失敗しました: {str(e)}</div>'
            
    def translate_pdf(self, pdf_path: str, output_html_path: str, max_pages: int = 20) -> bool:
        """
        PDF全体を翻訳してHTMLファイルを生成
        
        Args:
            pdf_path: 入力PDFのパス
            output_html_path: 出力HTMLのパス
            max_pages: 翻訳する最大ページ数
            
        Returns:
            成功したかどうか
        """
        logger.info(f"Starting PDF translation: {pdf_path}")
        
        try:
            # PDFを開く
            doc = fitz.open(pdf_path)
            total_pages = min(len(doc), max_pages)
            
            # HTML部品を集める
            html_parts = []
            
            # CSSスタイル
            css_style = """
<style>
    body {
        font-family: 'Noto Sans JP', sans-serif;
        line-height: 1.8;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .page {
        background: white;
        margin: 20px 0;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .page h3 {
        color: #333;
        border-bottom: 2px solid #007bff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .error {
        background-color: #fee;
        border-left: 4px solid #f44;
    }
    img {
        max-width: 100%;
        height: auto;
        margin: 20px 0;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f8f9fa;
    }
    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 4px;
        overflow-x: auto;
    }
    .math {
        font-family: 'Times New Roman', serif;
        font-style: italic;
    }
</style>
"""
            
            # HTMLヘッダー
            html_header = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>翻訳論文 - {os.path.basename(pdf_path)}</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
    {css_style}
</head>
<body>
    <h1>翻訳論文</h1>
    <p>元ファイル: {os.path.basename(pdf_path)}</p>
    <p>翻訳日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
    <hr>
"""
            
            html_parts.append(html_header)
            
            # 各ページを処理
            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                try:
                    page = doc[page_num]
                    content = self.extract_page_content(page)
                    
                    # 翻訳
                    translated_html = self.translate_page_content(content, page_num + 1)
                    html_parts.append(translated_html)
                    
                    # 画像をBase64エンコードして埋め込む
                    if content['images']:
                        html_parts.append('<div class="page-images">')
                        for idx, img in enumerate(content['images']):
                            # PIL ImageをBase64に変換
                            import base64
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            html_parts.append(f'''
                                <figure>
                                    <img src="data:image/png;base64,{img_base64}" alt="図{page_num + 1}-{idx + 1}">
                                    <figcaption>図{page_num + 1}-{idx + 1}</figcaption>
                                </figure>
                            ''')
                        html_parts.append('</div>')
                    
                    # API制限対策
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    html_parts.append(f'<div class="page error">ページ {page_num + 1} の処理中にエラーが発生しました</div>')
            
            # HTMLフッター
            html_footer = """
</body>
</html>
"""
            html_parts.append(html_footer)
            
            # ファイルに保存
            os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))
                
            logger.info(f"Translation completed: {output_html_path}")
            doc.close()
            return True
            
        except Exception as e:
            logger.error(f"PDF translation failed: {e}")
            return False
            
    def translate_from_url(self, url: str, output_html_path: str, max_pages: int = 20) -> bool:
        """
        URLから直接翻訳
        
        Args:
            url: PDFのURL
            output_html_path: 出力HTMLのパス
            max_pages: 翻訳する最大ページ数
            
        Returns:
            成功したかどうか
        """
        # PDFをダウンロード
        pdf_path = self.download_pdf(url)
        
        try:
            # 翻訳実行
            result = self.translate_pdf(pdf_path, output_html_path, max_pages)
            
            # 一時ファイルを削除
            if os.path.exists(pdf_path) and pdf_path.startswith("temp/"):
                os.remove(pdf_path)
                
            return result
            
        except Exception as e:
            logger.error(f"Translation from URL failed: {e}")
            # クリーンアップ
            if os.path.exists(pdf_path) and pdf_path.startswith("temp/"):
                os.remove(pdf_path)
            return False