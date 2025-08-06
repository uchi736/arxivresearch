"""
Simple PDF processing for Gemini long-context analysis

This module provides straightforward PDF text extraction and section splitting
without the complexity of RAG systems.
"""

import re
import requests
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
try:
    import pypdf as PyPDF2
except ImportError:
    import PyPDF2
import tiktoken

from src.core.models import PaperMetadata


class SimplePDFProcessor:
    """Simple PDF processor for text extraction and section identification"""
    
    # Section patterns for common academic paper structures
    SECTION_PATTERNS = [
        (r'(?i)^\s*abstract\s*$', 'abstract'),
        (r'(?i)^\s*(?:\d+\.?\s*)?introduction\s*$', 'introduction'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:related\s*work|background|prior\s*work|literature\s*review)\s*$', 'related_work'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:method|methods|methodology|approach|proposed\s*method|our\s*approach)\s*$', 'methods'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:experiment|experiments|experimental\s*setup|evaluation)\s*$', 'experiments'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:result|results|findings|experimental\s*results)\s*$', 'results'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:discussion|analysis|ablation\s*study)\s*$', 'discussion'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:conclusion|conclusions|summary|future\s*work)\s*$', 'conclusion'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:limitation|limitations)\s*$', 'limitations'),
        (r'(?i)^\s*(?:\d+\.?\s*)?(?:appendix|appendices|supplementary)\s*$', 'appendix'),
    ]
    
    def __init__(self):
        """Initialize the PDF processor"""
        # Initialize tokenizer for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def download_pdf(self, pdf_url: str, timeout: int = 30) -> Optional[bytes]:
        """
        Download PDF from URL
        
        Args:
            pdf_url: URL of the PDF
            timeout: Download timeout in seconds
            
        Returns:
            PDF content as bytes or None if failed
        """
        try:
            print(f"  Downloading PDF from {pdf_url}")
            response = requests.get(pdf_url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"  Error downloading PDF: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> Tuple[str, List[str]]:
        """
        Extract text from PDF content
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Tuple of (full_text, list_of_page_texts)
        """
        full_text = ""
        page_texts = []
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"  Extracting text from {num_pages} pages")
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        page_texts.append(page_text)
                        full_text += page_text + "\n\n"
                    except Exception as e:
                        print(f"  Warning: Failed to extract page {page_num + 1}: {e}")
                        page_texts.append("")
            
            os.unlink(tmp_file_path)
            
        except Exception as e:
            print(f"  Error extracting text from PDF: {e}")
            return "", []
        
        return full_text, page_texts
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split text into sections based on common academic paper structure
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        current_section = "header"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if this line is a section header
            section_matched = False
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    section_matched = True
                    break
            
            # If not a section header, add to current section
            if not section_matched and line.strip():
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no abstract was found but we have header content, try to extract it
        if 'abstract' not in sections and 'header' in sections:
            abstract = self._extract_abstract_from_header(sections['header'])
            if abstract:
                sections['abstract'] = abstract
        
        return sections
    
    def _extract_abstract_from_header(self, header_text: str) -> Optional[str]:
        """
        Try to extract abstract from header section
        
        Args:
            header_text: Text from the header section
            
        Returns:
            Abstract text if found, None otherwise
        """
        # Look for abstract keyword
        lines = header_text.split('\n')
        abstract_start = -1
        
        for i, line in enumerate(lines):
            if re.search(r'(?i)abstract', line):
                abstract_start = i + 1
                break
        
        if abstract_start >= 0 and abstract_start < len(lines):
            # Find where abstract ends (usually at next section or keywords)
            abstract_lines = []
            for i in range(abstract_start, len(lines)):
                line = lines[i].strip()
                if re.search(r'(?i)^(keywords|introduction|1\.|I\.)', line):
                    break
                if line:
                    abstract_lines.append(line)
            
            if abstract_lines:
                return ' '.join(abstract_lines)
        
        return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def process_paper(self, pdf_url: str) -> Optional[Dict]:
        """
        Process a paper from PDF URL to structured sections with token counts
        
        Args:
            pdf_url: URL of the paper PDF
            
        Returns:
            Dictionary with sections and metadata, or None if failed
        """
        # Download PDF
        pdf_content = self.download_pdf(pdf_url)
        if not pdf_content:
            return None
        
        # Extract text
        full_text, page_texts = self.extract_text_from_pdf(pdf_content)
        if not full_text:
            return None
        
        # Split into sections
        sections = self.split_into_sections(full_text)
        
        # Count tokens for each section
        section_tokens = {}
        total_tokens = 0
        
        for section_name, content in sections.items():
            tokens = self.count_tokens(content)
            section_tokens[section_name] = tokens
            total_tokens += tokens
        
        # Create summary of sections
        section_summary = []
        for section_name, content in sections.items():
            preview = content[:200] + "..." if len(content) > 200 else content
            section_summary.append({
                "name": section_name,
                "tokens": section_tokens[section_name],
                "preview": preview
            })
        
        return {
            "full_text": full_text,
            "sections": sections,
            "section_tokens": section_tokens,
            "total_tokens": total_tokens,
            "num_pages": len(page_texts),
            "section_summary": section_summary,
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    def prioritize_sections(self, sections: Dict[str, str], max_tokens: int) -> Dict[str, str]:
        """
        Prioritize sections to fit within token limit
        
        Args:
            sections: Dictionary of section name to content
            max_tokens: Maximum token limit
            
        Returns:
            Prioritized sections that fit within token limit
        """
        # Priority order for sections
        priority_order = [
            'abstract',
            'introduction', 
            'methods',
            'results',
            'conclusion',
            'experiments',
            'discussion',
            'related_work',
            'limitations',
            'header',
            'appendix'
        ]
        
        prioritized = {}
        current_tokens = 0
        
        for section_name in priority_order:
            if section_name in sections:
                section_tokens = self.count_tokens(sections[section_name])
                if current_tokens + section_tokens <= max_tokens:
                    prioritized[section_name] = sections[section_name]
                    current_tokens += section_tokens
                else:
                    # Try to include partial section
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 100:  # Only include if meaningful
                        truncated = self._truncate_to_tokens(
                            sections[section_name], 
                            remaining_tokens
                        )
                        prioritized[section_name] = truncated + "\n[TRUNCATED]"
                    break
        
        return prioritized
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Binary search for the right truncation point
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def get_section_for_ochiai_item(self, sections: Dict[str, str], ochiai_item: str) -> str:
        """
        Get relevant sections for a specific Ochiai format item
        
        Args:
            sections: Dictionary of sections
            ochiai_item: Ochiai format item name
            
        Returns:
            Concatenated relevant sections
        """
        # Mapping of Ochiai items to relevant sections
        ochiai_mapping = {
            'what_is_it': ['abstract', 'introduction'],
            'comparison_with_prior_work': ['related_work', 'introduction'],
            'key_technique': ['methods', 'approach'],
            'validation_method': ['experiments', 'evaluation'],
            'experimental_results': ['results', 'experiments'],
            'discussion_points': ['discussion', 'limitations'],
            'implementation_details': ['methods', 'appendix'],
        }
        
        relevant_sections = ochiai_mapping.get(ochiai_item, [])
        combined_text = []
        
        for section_name in relevant_sections:
            if section_name in sections:
                combined_text.append(f"[{section_name.upper()}]\n{sections[section_name]}")
        
        return '\n\n'.join(combined_text) if combined_text else ""