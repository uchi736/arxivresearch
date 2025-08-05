"""
arXiv search and paper retrieval functionality

This module handles searching arXiv API and retrieving paper metadata.
"""

from typing import List, Optional
import arxiv
from src.core.models import SearchQuery, PaperMetadata


def search_arxiv_papers(query: SearchQuery) -> List[PaperMetadata]:
    """
    Search arXiv API for papers with improved query construction
    
    Args:
        query: SearchQuery object with search parameters
        
    Returns:
        List of PaperMetadata objects
    """
    # Build search query with more targeted approach
    keyword_queries = []
    for kw in query.keywords:
        # Search in title AND abstract (more targeted than 'all:')
        keyword_queries.append(f'(ti:"{kw}" OR abs:"{kw}")')
    
    # Use OR between keywords for broader results
    search_query = " OR ".join(keyword_queries)
    
    # Add category constraint
    if query.category:
        search_query += f" AND cat:{query.category}"
    else:
        # For AI agent queries, default to relevant CS categories
        if any("agent" in kw.lower() or "AI" in kw or "evaluation" in kw.lower() 
               for kw in query.keywords):
            ai_categories = ["cs.AI", "cs.LG", "cs.CL", "cs.MA", "cs.RO", "cs.HC"]
            category_filter = " OR ".join([f"cat:{cat}" for cat in ai_categories])
            search_query += f" AND ({category_filter})"

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