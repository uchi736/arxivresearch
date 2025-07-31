"""
PDF processing and text extraction utilities

This module handles PDF download, text extraction, and page processing.
"""

import os
import requests
import tempfile
from typing import Optional, Dict, Any, List, Tuple
import PyPDF2
import pdfplumber


def extract_fulltext_advanced(pdf_url: str) -> Optional[Dict[str, Any]]:
    """
    Advanced PDF full text extraction
    
    Args:
        pdf_url: URL of the PDF to download and extract
        
    Returns:
        Dictionary containing full text, page texts, and metadata
    """
    print(f"  PDFダウンロード中: {pdf_url}")
    
    # Download PDF
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        pdf_content = response.content
    except Exception as e:
        print(f"  ダウンロードエラー: {e}")
        return None
    
    # Extract text from PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name
    
    text = ""
    page_texts = []
    
    try:
        # Try pdfplumber first for better extraction
        with pdfplumber.open(tmp_file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    page_texts.append((page_num + 1, page_text))
    except:
        # Fallback to PyPDF2
        try:
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    page_texts.append((page_num + 1, page_text))
        except Exception as e:
            print(f"  テキスト抽出エラー: {e}")
            os.unlink(tmp_file_path)
            return None
    
    os.unlink(tmp_file_path)
    
    if not text:
        return None
    
    return {
        "full_text": text,
        "page_texts": page_texts,
        "num_pages": len(page_texts)
    }


def update_chunk_page_numbers(
    chunks: List[Any], 
    page_texts: List[Tuple[int, str]]
) -> None:
    """
    Update page numbers for chunks based on text location
    
    Args:
        chunks: List of DocumentChunk objects
        page_texts: List of (page_num, text) tuples
    """
    for chunk in chunks:
        # Find which page contains this chunk
        chunk_start = chunk.text[:100] if len(chunk.text) > 100 else chunk.text
        for page_num, page_text in page_texts:
            if chunk_start in page_text:
                chunk.page_num = page_num
                break