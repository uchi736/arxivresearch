"""
arXiv search and paper retrieval functionality

This module handles searching arXiv API and retrieving paper metadata.
"""

from typing import List, Optional
import arxiv
from src.core.models import SearchQuery, PaperMetadata


def search_arxiv_papers(query: SearchQuery) -> List[PaperMetadata]:
    """
    Search arXiv API for papers
    
    Args:
        query: SearchQuery object with search parameters
        
    Returns:
        List of PaperMetadata objects
    """
    # Build search query
    search_query = " AND ".join([f'all:"{kw}"' for kw in query.keywords])
    if query.category:
        search_query += f" AND cat:{query.category}"

    # Map sort criteria
    sort_by_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }

    # Execute search
    search = arxiv.Search(
        query=search_query,
        max_results=query.max_results,
        sort_by=sort_by_map[query.sort_by],
    )

    results = []
    for result in search.results():
        paper_meta = PaperMetadata(
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            arxiv_id=result.entry_id.split('/')[-1],
            pdf_url=result.pdf_url,
            published_date=result.published.isoformat(),
            categories=result.categories,
        )
        results.append(paper_meta)
    
    return results