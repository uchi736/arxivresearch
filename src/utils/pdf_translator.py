import fitz  # PyMuPDF
import os
import time
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from src.core.config import create_llm_model
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader
from PIL import Image
import io

load_dotenv()

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

class PDFTranslatorWithReportLab:
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initializes the PDFTranslator with a language model.
        """
        self.llm = create_llm_model(model_name=model_name, temperature=0)
        self.rate_limiter = RateLimiter(calls_per_minute=15)
        os.makedirs("outputs", exist_ok=True)
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

    def translate_page_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts in a single API call"""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip() and len(text.strip()) > 2]
        if not valid_texts:
            return [""] * len(texts)
        
        # Create batch prompt
        combined_text = "\n---SEP---\n".join([f"[{i}]: {text}" for i, text in valid_texts])
        
        # Limit text length to avoid token limits
        if len(combined_text) > 4000:
            # Split into smaller batches
            mid = len(valid_texts) // 2
            first_half = [text for _, text in valid_texts[:mid]]
            second_half = [text for _, text in valid_texts[mid:]]
            
            first_translations = self.translate_page_batch(first_half)
            second_translations = self.translate_page_batch(second_half)
            
            # Reconstruct full result
            result = [""] * len(texts)
            for j, (orig_i, _) in enumerate(valid_texts[:mid]):
                result[orig_i] = first_translations[j] if j < len(first_translations) else ""
            for j, (orig_i, _) in enumerate(valid_texts[mid:]):
                result[orig_i] = second_translations[j] if j < len(second_translations) else ""
            
            return result
        
        prompt = f"""以下のテキストを日本語に翻訳してください。番号を保持して出力してください。

{combined_text}

出力形式:
[0]: 翻訳されたテキスト
[1]: 翻訳されたテキスト
..."""

        try:
            self.rate_limiter.wait()
            response = self.llm.invoke(prompt)
            self.rate_limiter.log_call()
            
            # Parse response
            translations = self._parse_batch_response(response.content, len(valid_texts))
            
            # Map back to original positions
            result = [""] * len(texts)
            for j, (orig_i, _) in enumerate(valid_texts):
                result[orig_i] = translations[j] if j < len(translations) else ""
            
            return result
            
        except Exception as e:
            print(f"    - Batch translation failed: {e}")
            # Fallback to individual translation
            return self._fallback_individual_translation([text for _, text in valid_texts], texts)
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """Parse batch translation response"""
        lines = response.strip().split('\n')
        translations = []
        
        for line in lines:
            if line.strip().startswith('[') and ']:' in line:
                try:
                    # Extract text after [N]:
                    _, text = line.split(']:', 1)
                    translations.append(text.strip())
                except ValueError:
                    continue
        
        # Ensure we have the right number of translations
        while len(translations) < expected_count:
            translations.append("")
        
        return translations[:expected_count]
    
    def _fallback_individual_translation(self, valid_texts: List[str], all_texts: List[str]) -> List[str]:
        """Fallback to individual translation if batch fails"""
        print("    - Falling back to individual translations...")
        result = [""] * len(all_texts)
        
        for i, text in enumerate(all_texts):
            if text.strip() and len(text.strip()) > 2:
                try:
                    self.rate_limiter.wait()
                    translated = self.llm.invoke(f"Translate to Japanese: '{text}'")
                    result[i] = translated.content
                    self.rate_limiter.log_call()
                except Exception as e:
                    print(f"    - Individual translation failed for text {i}: {e}")
                    result[i] = text  # Keep original if translation fails
        
        return result

    def translate_pdf(self, pdf_path, output_filename):
        """
        Translates PDF text content using PyMuPDF for extraction and ReportLab for generation.
        """
        doc = fitz.open(pdf_path)
        output_path = os.path.join("outputs", output_filename)
        
        c = canvas.Canvas(output_path, pagesize=letter)

        print(f"Processing {len(doc)} pages...")

        for page_num, page in enumerate(doc):
            print(f"  - Processing page {page_num + 1}/{len(doc)}")
            
            # Set page size in canvas
            page_width, page_height = page.rect.width, page.rect.height
            c.setPageSize((page_width, page_height))

            # 1. Extract and draw images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                img_rects = page.get_image_rects(img)
                if img_rects:
                    img_rect = img_rects[0]
                    # Convert PyMuPDF rect to ReportLab coordinates (bottom-left origin)
                    x, y, _, _ = img_rect
                    img_height = img_rect.height
                    y_rl = page_height - y - img_height
                    
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    c.drawImage(ImageReader(pil_img), x, y_rl, width=img_rect.width, height=img_height)

            # 2. Extract, translate, and draw text (batch processing)
            texts = []
            text_positions = []
            blocks = page.get_text("dict")["blocks"]
            
            # Collect all text spans first
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            rect = fitz.Rect(span["bbox"])
                            
                            texts.append(text)
                            text_positions.append({
                                'rect': rect,
                                'size': span["size"],
                                'text': text
                            })
            
            # Batch translate all texts for this page
            if texts:
                print(f"    - Translating {len(texts)} text spans in batch...")
                translated_texts = self.translate_page_batch(texts)
                
                # Draw translated texts
                for i, (translated, pos_info) in enumerate(zip(translated_texts, text_positions)):
                    if translated and translated.strip():
                        try:
                            # Use CID font for Japanese
                            c.setFont('HeiseiMin-W3', pos_info['size'])
                            # Convert coordinates
                            x, y = pos_info['rect'].bottom_left
                            y_rl = page_height - y
                            c.drawString(x, y_rl, translated)
                        except Exception as e:
                            print(f"    - Could not draw text: '{pos_info['text'][:50]}...'. Error: {e}")

            c.showPage() # Finalize the current page and start a new one

        c.save()
        doc.close()
        print(f"\nTranslation complete. Saved to {output_path}")
        return output_path

if __name__ == '__main__':
    if not os.path.exists("sample.pdf"):
        print("Creating a sample PDF for testing...")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "This is a test document.", fontsize=12)
        page.insert_text((50, 100), "This paper discusses the impact of AI on modern research.", fontsize=12)
        page.draw_rect(fitz.Rect(50, 120, 250, 220), color=(0,0,1), fill=(0.9, 0.9, 1))
        page.insert_text((60, 135), "Figure 1: A conceptual diagram.", fontsize=10)
        doc.save("sample.pdf")
        doc.close()
        print("Sample PDF created.")

    translator = PDFTranslatorWithReportLab()
    input_pdf = "sample.pdf"
    output_pdf_filename = "translated_sample_reportlab.pdf"
    
    print(f"\nStarting translation for '{input_pdf}'...")
    translator.translate_pdf(input_pdf, output_pdf_filename)
