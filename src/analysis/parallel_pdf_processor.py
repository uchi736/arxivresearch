"""
Parallel PDF processing with concurrent downloads and extraction

Optimized version of SimplePDFProcessor with parallel processing capabilities.
"""

import concurrent.futures
from typing import List, Dict, Optional, Tuple
import time
from src.analysis.simple_pdf_processor import SimplePDFProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelPDFProcessor:
    """PDF processor with parallel processing capabilities"""
    
    def __init__(self, max_workers: int = 3):
        """
        Initialize parallel PDF processor
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.processor = SimplePDFProcessor()
    
    def process_papers_parallel(
        self, 
        papers: List[Dict],
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Process multiple papers in parallel
        
        Args:
            papers: List of paper dictionaries with 'pdf_url' and metadata
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of processed paper data dictionaries
        """
        start_time = time.time()
        results = []
        completed = 0
        
        logger.info(f"Starting parallel PDF processing for {len(papers)} papers with {self.max_workers} workers")
        
        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all PDF processing tasks
            future_to_paper = {
                executor.submit(self._process_single_paper, paper): paper 
                for paper in papers
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.debug(f"Successfully processed: {paper.get('title', 'Unknown')[:50]}...")
                    else:
                        logger.warning(f"Failed to process: {paper.get('title', 'Unknown')[:50]}...")
                        
                except Exception as e:
                    logger.error(f"Error processing {paper.get('title', 'Unknown')}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(papers))
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel PDF processing completed in {elapsed:.1f}s. Successfully processed {len(results)}/{len(papers)} papers")
        
        return results
    
    def _process_single_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Process a single paper (worker function)
        
        Args:
            paper: Paper dictionary with metadata
            
        Returns:
            Processed paper data or None if failed
        """
        try:
            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                logger.error(f"No PDF URL for paper: {paper.get('title', 'Unknown')}")
                return None
            
            # Process PDF
            pdf_data = self.processor.process_paper(pdf_url)
            
            if pdf_data:
                # Merge with paper metadata
                result = {
                    'arxiv_id': paper.get('arxiv_id'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'abstract': paper.get('abstract'),
                    'pdf_data': pdf_data,
                    'relevance_score': paper.get('relevance_score', 0.0),
                    'processing_status': 'success'
                }
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Exception in _process_single_paper: {e}")
            return None
    
    def download_pdfs_parallel(
        self,
        pdf_urls: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, Optional[bytes]]]:
        """
        Download multiple PDFs in parallel
        
        Args:
            pdf_urls: List of PDF URLs
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of tuples (url, pdf_content)
        """
        results = []
        completed = 0
        
        logger.info(f"Starting parallel download of {len(pdf_urls)} PDFs")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.processor.download_pdf, url): url 
                for url in pdf_urls
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    pdf_content = future.result()
                    results.append((url, pdf_content))
                    if pdf_content:
                        logger.debug(f"Downloaded: {url}")
                    else:
                        logger.warning(f"Failed to download: {url}")
                        
                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")
                    results.append((url, None))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(pdf_urls))
        
        return results
    
    def extract_sections_parallel(
        self,
        papers_with_content: List[Tuple[Dict, bytes]],
        max_tokens_per_paper: int = 10000
    ) -> List[Dict]:
        """
        Extract sections from multiple PDFs in parallel
        
        Args:
            papers_with_content: List of tuples (paper_metadata, pdf_content)
            max_tokens_per_paper: Maximum tokens to keep per paper
            
        Returns:
            List of processed papers with sections
        """
        results = []
        
        def process_pdf_content(paper_data):
            paper, pdf_content = paper_data
            try:
                # Extract text
                full_text, page_texts = self.processor.extract_text_from_pdf(pdf_content)
                if not full_text:
                    return None
                
                # Split into sections
                sections = self.processor.split_into_sections(full_text)
                
                # Prioritize sections to fit token limit
                prioritized_sections = self.processor.prioritize_sections(
                    sections, 
                    max_tokens_per_paper
                )
                
                # Count tokens
                total_tokens = sum(
                    self.processor.count_tokens(content) 
                    for content in prioritized_sections.values()
                )
                
                return {
                    'paper': paper,
                    'sections': prioritized_sections,
                    'total_tokens': total_tokens,
                    'num_pages': len(page_texts)
                }
            except Exception as e:
                logger.error(f"Error processing PDF content: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(process_pdf_content, data) 
                for data in papers_with_content
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in section extraction: {e}")
        
        return results


# Helper function for integration with existing code
def process_papers_batch(
    papers: List[Dict], 
    max_workers: int = 3,
    max_tokens_per_paper: int = 10000,
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Batch process papers with parallel PDF processing
    
    Args:
        papers: List of paper dictionaries
        max_workers: Number of parallel workers
        max_tokens_per_paper: Token limit per paper
        progress_callback: Optional progress callback
        
    Returns:
        List of processed papers
    """
    processor = ParallelPDFProcessor(max_workers=max_workers)
    
    # Process papers in parallel
    processed_papers = processor.process_papers_parallel(
        papers, 
        progress_callback=progress_callback
    )
    
    # Filter and sort by relevance score
    successful_papers = [p for p in processed_papers if p and p.get('processing_status') == 'success']
    successful_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
    
    return successful_papers