"""
Unified paper processor supporting both HTML and PDF formats
"""

from typing import Optional, Dict
from .simple_pdf_processor import SimplePDFProcessor
from .html_processor import ArXivHTMLProcessor
import logging

logger = logging.getLogger(__name__)


class UnifiedPaperProcessor:
    """
    Unified processor that can handle both HTML and PDF formats
    Automatically falls back to PDF if HTML processing fails
    """
    
    def __init__(self, prefer_html: bool = True):
        """
        Initialize processor
        
        Args:
            prefer_html: Whether to prefer HTML format over PDF (faster but less reliable)
        """
        self.prefer_html = prefer_html
        self.html_processor = ArXivHTMLProcessor()
        self.pdf_processor = SimplePDFProcessor()
    
    def arxiv_id_to_urls(self, arxiv_id: str) -> Dict[str, str]:
        """Convert arXiv ID to both HTML and PDF URLs"""
        return {
            'html': f'https://arxiv.org/html/{arxiv_id}',
            'pdf': f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        }
    
    def process_paper(self, url_or_id: str, format_preference: str = "auto") -> Optional[Dict]:
        """
        Process a paper using the best available format
        
        Args:
            url_or_id: arXiv URL or ID
            format_preference: "auto", "html", or "pdf"
            
        Returns:
            Dictionary with extracted content or None if failed
        """
        # Extract arXiv ID from URL if needed
        if 'arxiv.org' in url_or_id:
            arxiv_id = url_or_id.split('/')[-1].replace('.pdf', '')
        else:
            arxiv_id = url_or_id
        
        urls = self.arxiv_id_to_urls(arxiv_id)
        
        # Determine processing order based on preference
        if format_preference == "html":
            formats_to_try = [("html", urls['html'])]
        elif format_preference == "pdf":
            formats_to_try = [("pdf", urls['pdf'])]
        else:  # auto
            if self.prefer_html:
                formats_to_try = [("html", urls['html']), ("pdf", urls['pdf'])]
            else:
                formats_to_try = [("pdf", urls['pdf']), ("html", urls['html'])]
        
        # Try each format in order
        for format_type, url in formats_to_try:
            try:
                if format_type == "html":
                    result = self.html_processor.process_paper(url)
                else:
                    result = self.pdf_processor.process_paper(url)
                
                if result:
                    result['format_used'] = format_type
                    result['arxiv_id'] = arxiv_id
                    # Ensure consistent key names
                    if 'text' in result and 'full_text' not in result:
                        result['full_text'] = result['text']
                    logger.info(f"Successfully processed using {format_type.upper()} format")
                    return result
                    
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    logger.info(f"HTML format not available for {arxiv_id}, trying PDF fallback...")
                else:
                    logger.warning(f"Failed to process with {format_type} format: {e}")
                continue
        
        logger.error(f"Failed to process paper {arxiv_id} with any format")
        return None
    
    def get_paper_urls(self, arxiv_id: str) -> Dict[str, str]:
        """Get all available URLs for a paper"""
        return {
            'pdf': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
            'html': f'https://arxiv.org/html/{arxiv_id}',
            'abs': f'https://arxiv.org/abs/{arxiv_id}',
            'source': f'https://arxiv.org/src/{arxiv_id}'
        }