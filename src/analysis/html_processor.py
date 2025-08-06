"""
HTML-based paper processor for arXiv papers
Provides faster text extraction using arXiv HTML format
"""

import requests
import re
from typing import Optional, Dict
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class ArXivHTMLProcessor:
    """Process arXiv papers using HTML format for faster text extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_html(self, html_url: str, timeout: int = 30) -> Optional[str]:
        """
        Download HTML content from arXiv
        
        Args:
            html_url: URL of the HTML page
            timeout: Request timeout in seconds
            
        Returns:
            HTML content as string or None if failed
        """
        try:
            print(f"  Downloading HTML from {html_url}")
            response = self.session.get(html_url, timeout=timeout)
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading HTML: {html_url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading HTML: {e}")
            return None
    
    def extract_text_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract clean text from arXiv HTML content
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Extracted text or None if failed
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Find main content (arXiv HTML has specific structure)
            main_content = soup.find('div', class_='ltx_document')
            if not main_content:
                # Fallback to body content
                main_content = soup.find('body')
            
            if main_content:
                text = main_content.get_text()
                # Clean up text
                text = re.sub(r'\n+', '\n', text)
                text = re.sub(r' +', ' ', text)
                return text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return None
    
    def process_paper(self, html_url: str) -> Optional[Dict]:
        """
        Process an arXiv paper using HTML format
        
        Args:
            html_url: URL of the paper HTML
            
        Returns:
            Dictionary with extracted content or None if failed
        """
        # Download HTML content
        html_content = self.download_html(html_url)
        if not html_content:
            return None
        
        # Extract text
        text = self.extract_text_from_html(html_content)
        if not text:
            return None
        
        # Estimate page count (rough approximation)
        estimated_pages = max(1, len(text) // 3000)  # ~3000 chars per page
        
        return {
            'text': text,
            'page_count': estimated_pages,
            'source_format': 'html',
            'url': html_url
        }