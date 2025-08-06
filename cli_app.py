#!/usr/bin/env python
"""
CLI Application for arXiv Research Agent

Usage:
    python cli_app.py "your search query" [options]
    
Options:
    --depth shallow|moderate|deep  Analysis depth (default: moderate)
    --workflow standard  Workflow type (default: standard)
    --papers N  Number of papers to analyze (default: 5)
    --output FILE  Save report to file (default: print to console)
"""

import sys
import os
import json
from datetime import datetime
import time
import logging
import argparse
import re
import requests
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# arXiv URL normalization functions
ARXIV_ID_RE = re.compile(r'(\d{4}\.\d{4,5})(v\d+)?')
ARXIV_OLD_ID_RE = re.compile(r'([a-z\-]+(?:\.[A-Z]{2})?/\d{7})', re.I)

def to_arxiv_id(url_or_id: str) -> str:
    """Extracts a clean arXiv ID from a URL or string, supporting new, old, and versioned formats."""
    match = ARXIV_ID_RE.search(url_or_id) or ARXIV_OLD_ID_RE.search(url_or_id)
    if match:
        return ''.join(filter(None, match.groups()))
    raise ValueError(f"Could not extract a valid arXiv ID from: {url_or_id}")

def to_pdf_url(url_or_id: str) -> str:
    """Converts any arXiv URL or ID to its canonical PDF URL."""
    return f'https://arxiv.org/pdf/{to_arxiv_id(url_or_id)}.pdf'

def to_html_url(url_or_id: str) -> str:
    """Converts any arXiv URL or ID to its HTML URL."""
    return f'https://arxiv.org/html/{to_arxiv_id(url_or_id)}'

def to_arxiv_url(url_or_id: str, format_type: str = "pdf") -> str:
    """
    Converts any arXiv URL or ID to the specified format URL.
    
    Args:
        url_or_id: arXiv URL or ID
        format_type: "pdf" or "html"
    
    Returns:
        Standardized HTTPS arXiv URL
    """
    arxiv_id = to_arxiv_id(url_or_id)
    
    if format_type == "html":
        return f'https://arxiv.org/html/{arxiv_id}'
    elif format_type == "pdf":
        return f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    else:
        raise ValueError(f"Unsupported format: {format_type}. Use 'pdf' or 'html'.")

def download_pdf_safe(url: str, out_path: str) -> bool:
    """
    Downloads a PDF from a URL with robust validation.
    Normalizes arXiv URLs and verifies Content-Type and PDF header.
    """
    try:
        if not url.startswith('http'):
            # Assume it's a local file path
            if os.path.exists(url):
                return True
            else:
                raise FileNotFoundError(f"Local file not found: {url}")

        pdf_url = to_pdf_url(url) if 'arxiv.org' in url else url
        logger.info(f"Downloading PDF from normalized URL: {pdf_url}")
        
        r = requests.get(pdf_url, stream=True, timeout=30)
        r.raise_for_status()
        
        # Robust Validation
        content_type = r.headers.get('content-type', '').lower()
        first_chunk = r.raw.read(1024) # Read the first 1KB for header check
        
        if 'application/pdf' not in content_type or not first_chunk.startswith(b'%PDF-'):
            raise RuntimeError(f"Validation failed: Not a valid PDF. URL: {pdf_url}, Content-Type: {content_type}")

        logger.info(f"Validated PDF successfully (Content-Type: {content_type})")
        
        # Download the verified file
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(first_chunk) # Write the chunk we already read
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
        
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        return False

# Delayed imports to prevent execution during module loading
# These will be imported inside functions where needed


class CLIProgressDisplay:
    """Simple progress display for CLI"""
    
    def __init__(self):
        self.last_message = ""
    
    def update(self, message: str, progress: Optional[float] = None):
        """Update progress display"""
        if progress is not None:
            bar_length = 30
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            print(f"\r[{bar}] {int(progress*100)}% - {message}", end="", flush=True)
        else:
            print(f"\r{message}", end="", flush=True)
        self.last_message = message
    
    def complete(self, message: str = "Complete"):
        """Complete current progress"""
        print(f"\r{self.last_message} - {message}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="arXiv Research Agent - AI-powered paper analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_app.py "transformer architecture"
  python cli_app.py "AIエージェントの評価" --depth deep --papers 10
  python cli_app.py "LLM fine-tuning" --output report.md
  python cli_app.py translate https://arxiv.org/pdf/2312.xxxxx.pdf
  python cli_app.py "query" --translate 1,3,5
        """
    )
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Main search command (default)
    search_parser = subparsers.add_parser('search', help='Search and analyze papers (default)', add_help=False)
    search_parser.add_argument("query", help="Search query (Japanese or English)")
    search_parser.add_argument("--depth", choices=["shallow", "moderate", "deep"], 
                       default="moderate", help="Analysis depth")
    search_parser.add_argument("--workflow", choices=["standard"], 
                       default="standard", help="Workflow type")
    search_parser.add_argument("--papers", type=int, default=5, 
                       help="Number of papers to analyze")
    search_parser.add_argument("--output", help="Output file for report (markdown)")
    search_parser.add_argument("--token-budget", type=int, default=30000,
                       help="Token budget for analysis")
    search_parser.add_argument("--translate", help="Translate specific papers after analysis (comma-separated indices, e.g., 1,3,5)")
    search_parser.add_argument("--skip-analyzed", action="store_true", help="Skip papers that have already been analyzed")
    search_parser.add_argument("--format", choices=["auto", "html", "pdf"], default="auto",
                       help="Paper processing format: auto (HTML first, PDF fallback), html (HTML only), pdf (PDF only)")
    
    # Translate command
    translate_parser = subparsers.add_parser('translate', help='Translate a PDF paper')
    translate_parser.add_argument("pdf", help="PDF URL or local file path")
    translate_parser.add_argument("--output", help="Output HTML file path", default=None)
    translate_parser.add_argument("--max-pages", type=int, default=20, help="Maximum pages to translate")
    translate_parser.add_argument("--academic", action="store_true", help="Use academic paper translation mode (better for complex papers)")
    translate_parser.add_argument("--html-only", action="store_true", help="Disable PDF fallback (HTML translation only)")
    
    # Registry command
    registry_parser = subparsers.add_parser('registry', help='Manage analyzed papers database')
    registry_parser.add_argument("--schema", choices=["min", "full", "legacy"], default="min", 
                                help="CSV schema: min (8 cols), full (20 cols), legacy (21 cols)")
    registry_subparsers = registry_parser.add_subparsers(dest='registry_action', help='Registry actions')
    
    # Registry list command
    list_parser = registry_subparsers.add_parser('list', help='List analyzed papers')
    list_parser.add_argument("--query", help="Filter by query used")
    list_parser.add_argument("--status", choices=["completed", "pending", "failed"], help="Filter by status")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of results")
    list_parser.add_argument("--recent", type=int, help="Show papers from last N days")
    
    # Registry search command
    search_registry_parser = registry_subparsers.add_parser('search', help='Search analyzed papers')
    search_registry_parser.add_argument("query", help="Search query")
    search_registry_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # Registry stats command
    stats_parser = registry_subparsers.add_parser('stats', help='Show registry statistics')
    stats_parser.add_argument("--days", type=int, default=30, help="Statistics for last N days")
    
    # Registry cleanup command
    cleanup_parser = registry_subparsers.add_parser('cleanup', help='Clean up old entries')
    cleanup_parser.add_argument("--days", type=int, default=90, help="Keep entries newer than N days")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    
    # Registry export command
    export_parser = registry_subparsers.add_parser('export', help='Export registry to Excel')
    export_parser.add_argument("--output", default="registry_export.xlsx", help="Output Excel file")
    
    # Registry backup command
    backup_parser = registry_subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument("--suffix", help="Backup suffix (default: timestamp)")
    
    # Registry reset command
    reset_parser = registry_subparsers.add_parser('reset', help='Reset database (DANGEROUS)')
    reset_parser.add_argument("--no-backup", action="store_true", help="Skip backup before reset")
    reset_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    # Registry restore command
    restore_parser = registry_subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument("backup_dir", help="Backup directory to restore from")
    
    # For backward compatibility, if no subcommand is provided, assume 'search'
    args = parser.parse_args()
    if args.command is None and len(sys.argv) > 1:
        # Re-parse as search command
        args = parser.parse_args(['search'] + sys.argv[1:])
    
    return args


def build_workflow(workflow_type: str):
    """Build the specified workflow"""
    if workflow_type != "standard":
        raise ValueError("Only 'standard' workflow is supported.")
    
    workflow_names = {
        "standard": "Standard (with RAG)"
    }
    
    print(f"\nBuilding workflow: {workflow_names[workflow_type]}")
    
    # Lazy import to prevent execution during module loading
    from src.core.workflow import build_advanced_workflow
    return build_advanced_workflow()


def print_header(query: str, args):
    """Print application header"""
    print("\n" + "="*70)
    print(" arXiv Research Agent")
    print("="*70)
    print(f"\nQuery: {query}")
    print(f"Analysis depth: {args.depth}")
    print(f"Papers to analyze: {args.papers}")
    print(f"Workflow: {args.workflow}")
    print(f"Token budget: {args.token_budget:,}")
    if args.output:
        print(f"Output file: {args.output}")
    print("\n" + "-"*70)


def display_results(result: dict, output_file: Optional[str] = None):
    """Display or save results"""
    # Print summary
    print("\n\n" + "="*70)
    print(" Analysis Results")
    print("="*70)
    
    # Summary statistics
    if result.get("found_papers"):
        print(f"\nPapers found: {len(result['found_papers'])}")
    
    if result.get("analyzed_papers"):
        print(f"Papers analyzed: {len(result['analyzed_papers'])}")
        
        # Show analyzed papers
        print("\nAnalyzed Papers:")
        for i, paper in enumerate(result['analyzed_papers'][:5], 1):
            metadata = paper.get('metadata', {})
            print(f"\n{i}. {metadata.get('title', 'Unknown')}")
            print(f"   Authors: {', '.join(metadata.get('authors', [])[:3])}")
            if paper.get('relevance_score'):
                print(f"   Relevance: {paper['relevance_score']:.1f}/10")
            if paper.get('analysis_type'):
                print(f"   Analysis: {paper['analysis_type']}")
    
    # Final report
    if result.get("final_report"):
        report = result["final_report"]
        
        if output_file:
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n[OK] Report saved to: {output_file}")
            print(f"   ({len(report):,} characters)")
        else:
            # Print to console
            print("\n" + "="*70)
            print(" Final Report")
            print("="*70)
            print(report)
    
    # Token usage
    total_tokens = sum(p.get('tokens_used', 0) for p in result.get('analyzed_papers', []))
    if total_tokens > 0:
        print(f"\nTotal tokens used: {total_tokens:,}")


def run_research(query: str, args):
    """Run the research workflow with registry integration"""
    # Lazy imports to prevent execution during module loading
    from src.registry import CSVPaperRegistry
    from src.core.models import AdvancedAgentState
    
    # Clear global state before starting new research
    logger.info("Starting new research - clearing global state")
    from src.workflow.nodes.base import clear_paper_memories
    from src.core.dependencies import clear_container_cache
    
    clear_paper_memories()
    clear_container_cache()
    
    progress = CLIProgressDisplay()
    registry = CSVPaperRegistry()
    
    try:
        # Build workflow
        workflow = build_workflow(args.workflow)
        
        # Create initial state
        initial_state = {
            "initial_query": query,
            "research_plan": None,
            "search_queries": [],
            "found_papers": [],
            "analyzed_papers": [],
            "final_report": "",
            "token_budget": args.token_budget,
            "analysis_mode": f"advanced_{args.depth}",
            "total_tokens_used": 0,
            "progress_tracker": None,
            "num_papers": args.papers
        }
        
        # Add registry integration flag
        initial_state["skip_analyzed"] = getattr(args, 'skip_analyzed', False)
        initial_state["registry"] = registry
        
        # Add paper format preference
        initial_state["paper_format"] = getattr(args, 'format', 'auto')
        
        print(f"\nStarting research...")
        if getattr(args, 'skip_analyzed', False):
            print(f"[INFO] Skip-analyzed mode enabled - will filter out analyzed papers")
        
        start_time = time.time()
        
        # Progress updates
        progress.update("Planning research...", 0.1)
        
        # Execute workflow with timeout monitoring
        logger.debug(f"Starting workflow.invoke at {datetime.now()}")
        invoke_start = time.time()
        
        try:
            result = workflow.invoke(initial_state)
            invoke_elapsed = time.time() - invoke_start
            logger.debug(f"workflow.invoke completed in {invoke_elapsed:.1f}s")
        except Exception as e:
            invoke_elapsed = time.time() - invoke_start
            logger.error(f"workflow.invoke failed after {invoke_elapsed:.1f}s: {e}")
            raise
        
        elapsed = time.time() - start_time
        progress.complete(f"Done in {elapsed:.1f}s")
        
        # Register analyzed papers in CSV registry
        analyzed_papers = result.get('analyzed_papers', [])
        if analyzed_papers:
            print(f"\n[INFO] Registering {len(analyzed_papers)} analyzed papers...")
            for paper in analyzed_papers:
                try:
                    # Add query information
                    paper['query_used'] = query
                    registry.register_analyzed_paper(paper)
                except Exception as e:
                    logger.warning(f"Failed to register paper {paper.get('metadata', {}).get('arxiv_id', 'Unknown')}: {e}")
            
            # Update search history
            found_papers = result.get('found_papers', [])
            papers_found = len(found_papers)
            papers_new = papers_found  # Will be updated if skip-analyzed was used
            papers_analyzed = len(analyzed_papers)
            
            if getattr(args, 'skip_analyzed', False):
                # Calculate how many were actually new
                original_found = result.get('original_papers_found', papers_found)
                papers_new = papers_found  # After filtering
                papers_found = original_found  # Original count
            
            registry.update_search_history(
                query=query,
                papers_found=papers_found,
                papers_new=papers_new, 
                papers_analyzed=papers_analyzed,
                execution_time=int(elapsed),
                analysis_mode=args.depth
            )
        
        # Display results
        display_results(result, args.output)
        
        return result  # Return the result dict instead of True
        
    except KeyboardInterrupt:
        print("\n\n[!] Research interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n\n[ERROR] {type(e).__name__}: {e}")
        logger.exception("Research failed")
        return None


def translate_paper(pdf_path: str, output_path: Optional[str] = None, max_pages: int = 20, academic_mode: bool = False, html_only: bool = False):
    """Translate a single paper"""
    try:
        # Check if it's an arXiv URL and try HTML translation first
        if "arxiv.org" in pdf_path:
            # Check if user wants academic mode
            if academic_mode:
                from src.translation.professional_arxiv_translator import ProfessionalArxivTranslator
                html_translator = ProfessionalArxivTranslator()
                print("[INFO] Using professional academic translator (UTF-8 optimized)")
            else:
                from src.translation.arxiv_html_translator import ArxivHTMLTranslator
                html_translator = ArxivHTMLTranslator()
            
            # Extract arXiv ID using proper normalization
            try:
                arxiv_id = to_arxiv_id(pdf_path)
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                
                print(f"\nChecking ar5iv availability for arXiv paper {arxiv_id}...")
                if html_translator.check_ar5iv_availability(arxiv_url):
                    print("[OK] Using fast ar5iv HTML translation")
                    
                    # Set output path for HTML translation
                    if output_path is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"outputs/translations/{arxiv_id}_{timestamp}.html"
                    
                    success = html_translator.translate_arxiv_paper(arxiv_url, output_path)
                    
                    if success:
                        print(f"[OK] Translation saved to: {output_path}")
                        return True
                    else:
                        print("[WARNING] HTML translation failed")
                        if html_only:
                            print("[INFO] HTML-only mode: skipping PDF fallback")
                            return False
                        print("[INFO] Falling back to PDF translation...")
                else:
                    print("[INFO] ar5iv not available")
                    if html_only:
                        print("[INFO] HTML-only mode: skipping PDF fallback")
                        return False
                    print("[INFO] Using PDF translation")
            except ValueError as e:
                print(f"[ERROR] Invalid arXiv URL: {e}")
                return False
        
        # Skip PDF fallback if html-only mode
        if html_only:
            print("[INFO] HTML-only mode: PDF translation disabled")
            return False
            
        # Fall back to PDF translation with proper URL normalization
        from src.translation import PDFTranslatorWithReportLab
        
        # Convert to proper PDF URL if it's an arXiv URL
        actual_pdf_path = pdf_path
        if "arxiv.org" in pdf_path:
            try:
                actual_pdf_path = to_pdf_url(pdf_path)
                print(f"[INFO] Normalized PDF URL: {actual_pdf_path}")
            except ValueError:
                print(f"[WARNING] Could not normalize arXiv URL, using as-is: {pdf_path}")
        
        print(f"\nTranslating paper: {actual_pdf_path}")
        
        # Set default output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if actual_pdf_path.startswith("http"):
                # Extract filename from URL properly
                try:
                    arxiv_id = to_arxiv_id(actual_pdf_path)
                    filename = arxiv_id
                except:
                    filename = actual_pdf_path.split('/')[-1].replace('.pdf', '')
            else:
                filename = os.path.basename(actual_pdf_path).replace('.pdf', '')
            output_path = f"outputs/translations/{filename}_{timestamp}.html"
        
        # Create translator
        translator = PDFTranslatorWithReportLab()
        
        # Translate with safe download
        if actual_pdf_path.startswith("http"):
            # Use safe download for URLs
            temp_pdf = f"temp/{to_arxiv_id(actual_pdf_path) if 'arxiv.org' in actual_pdf_path else 'downloaded'}.pdf"
            if download_pdf_safe(actual_pdf_path, temp_pdf):
                success = translator.translate_pdf(temp_pdf, output_path, max_pages)
                # Clean up temp file
                try:
                    os.remove(temp_pdf)
                except:
                    pass
            else:
                return False
        else:
            success = translator.translate_pdf(actual_pdf_path, output_path, max_pages)
        
        if success:
            print(f"[OK] Translation saved to: {output_path}")
        else:
            print("[ERROR] Translation failed")
        
        return success
        
    except ImportError as e:
        if "PyMuPDF" in str(e):
            print("[ERROR] PyMuPDF is required for translation. Install with: pip install PyMuPDF")
        else:
            print(f"[ERROR] Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return False


def translate_from_results(result: dict, indices: str):
    """Translate specific papers from analysis results"""
    try:
        # Parse indices
        paper_indices = [int(i.strip()) - 1 for i in indices.split(',')]
        
        analyzed_papers = result.get('analyzed_papers', [])
        if not analyzed_papers:
            print("[ERROR] No analyzed papers found")
            return
        
        print(f"\n\nTranslating selected papers...")
        print("="*70)
        
        for idx in paper_indices:
            if 0 <= idx < len(analyzed_papers):
                paper = analyzed_papers[idx]
                metadata = paper.get('metadata', {})
                pdf_link = metadata.get('pdf_link')
                
                if pdf_link:
                    print(f"\n[{idx+1}] {metadata.get('title', 'Unknown')}")
                    translate_paper(pdf_link)
                else:
                    print(f"\n[{idx+1}] No PDF link found for: {metadata.get('title', 'Unknown')}")
            else:
                print(f"\n[ERROR] Invalid index: {idx+1} (valid range: 1-{len(analyzed_papers)})")
                
    except Exception as e:
        print(f"[ERROR] Failed to translate papers: {e}")


def handle_registry_command(args):
    """Handle registry subcommands"""
    from src.registry import PaperRegistryConfig, CSVPaperRegistry
    
    # Create config with selected schema
    config = PaperRegistryConfig(schema_version=getattr(args, 'schema', 'min'))
    registry = CSVPaperRegistry(config)
    
    if args.registry_action == 'list':
        # List analyzed papers
        query_filter = args.query if hasattr(args, 'query') else None
        status_filter = args.status if hasattr(args, 'status') else None
        limit = args.limit if hasattr(args, 'limit') else 20
        
        df = registry.get_analyzed_papers(
            query_filter=query_filter,
            status_filter=status_filter,
            limit=limit
        )
        
        if df.empty:
            print("No analyzed papers found matching criteria")
            return
        
        print(f"\nAnalyzed Papers ({len(df)} found)")
        print("="*80)
        
        for idx, row in df.iterrows():
            print(f"\n{idx+1}. {row.get('title', 'Unknown')[:60]}...")
            print(f"   arXiv ID: {row.get('arxiv_id', 'Unknown')}")
            print(f"   Query: {row.get('query_used', 'Unknown')}")
            print(f"   Analyzed: {row.get('analyzed_at', 'Unknown')}")
            score = row.get('relevance_score', 0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            print(f"   Score: {score:.1f}/10")
            print(f"   Status: {row.get('status', 'Unknown')}")
    
    elif args.registry_action == 'search':
        # Search papers by query
        df = registry.get_analyzed_papers(query_filter=args.query, limit=args.limit)
        
        if df.empty:
            print(f"No papers found for query: {args.query}")
            return
        
        print(f"\nSearch Results for '{args.query}' ({len(df)} found)")
        print("="*80)
        
        for idx, row in df.iterrows():
            print(f"\n{idx+1}. {row.get('title', 'Unknown')}")
            summary = row.get('analysis_summary', 'No summary')
            if isinstance(summary, str) and len(summary) > 100:
                summary = summary[:100] + "..."
            print(f"   Summary: {summary}")
            score = row.get('relevance_score', 0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            print(f"   Score: {score:.1f}/10")
    
    elif args.registry_action == 'stats':
        # Show statistics
        stats = registry.get_search_statistics(days=args.days)
        info = registry.get_registry_info()
        
        print(f"\nRegistry Statistics (Last {args.days} days)")
        print("="*50)
        print(f"Total Papers: {info['total_papers']}")
        print(f"Completed: {info['completed_papers']}")
        print(f"Failed: {info['failed_papers']}")
        print(f"Total Searches: {stats['total_searches']}")
        print(f"Papers Found: {stats['total_papers_found']}")
        print(f"Papers Analyzed: {stats['total_papers_analyzed']}")
        print(f"Avg Execution Time: {stats['avg_execution_time']:.1f}s")
        
        if stats['recent_queries']:
            print(f"\nRecent Queries:")
            for query in stats['recent_queries'][-5:]:
                print(f"  - {query}")
    
    elif args.registry_action == 'cleanup':
        # Clean up old entries
        if args.dry_run:
            print(f"[DRY RUN] Would clean up entries older than {args.days} days")
            # TODO: Add dry-run logic to show what would be deleted
        else:
            print(f"Cleaning up entries older than {args.days} days...")
            registry.cleanup_old_entries(days=args.days)
            print("Cleanup completed")
    
    elif args.registry_action == 'export':
        # Export to Excel with enhanced formatting and Japanese headers by default
        format_name = getattr(args, 'format', 'enhanced')
        print(f"Exporting registry to {args.output} (format: {format_name})...")
        try:
            # Use enhanced formatting for full 21-column export
            use_formatting = format_name == 'enhanced'
            registry.export_to_excel(args.output, 
                                   use_japanese_headers=True,
                                   enhanced_formatting=use_formatting)
            
            # Show export summary
            if use_formatting:
                print(f"Export completed with enhanced formatting: {args.output}")
                print("Features: 21-column format, auto-width, header styling, freeze panes")
            else:
                print(f"Export completed (basic format): {args.output}")
                
        except ImportError as e:
            print(f"[ERROR] {e}")
            print(f"[SOLUTION] Run: pip install openpyxl")
            print(f"[ALTERNATIVE] Use CSV format instead:")
            csv_file = args.output.replace('.xlsx', '.csv')
            print(f"             python cli_app.py registry export --output {csv_file}")
            return 1
    
    elif args.registry_action == 'backup':
        # Create database backup
        suffix = getattr(args, 'suffix', None)
        backup_dir = registry.backup_database(backup_suffix=suffix)
        print(f"Database backup created: {backup_dir}")
    
    elif args.registry_action == 'reset':
        # Reset database with safety checks
        backup_first = not getattr(args, 'no_backup', False)
        force = getattr(args, 'force', False)
        
        # Show current stats before reset
        info = registry.get_registry_info()
        print(f"\n[WARNING] Database Reset")
        print(f"Current data: {info['total_papers']} papers, {info['total_searches']} searches")
        print(f"Files: {info['papers_file']}, {info['history_file']}")
        
        if not force:
            print(f"\nThis will permanently delete all data!")
            if backup_first:
                print("A backup will be created first.")
            else:
                print("NO BACKUP will be created (--no-backup specified).")
            
            confirm = input("\nType 'YES' to confirm reset: ")
            if confirm != 'YES':
                print("Reset cancelled.")
                return 0
        
        # Perform reset
        success = registry.reset_database(backup_first=backup_first, confirm=False)
        if success:
            print("Database reset completed successfully.")
        else:
            print("Database reset failed.")
            return 1
    
    elif args.registry_action == 'restore':
        # Restore from backup
        backup_dir = args.backup_dir
        
        if not os.path.exists(backup_dir):
            print(f"[ERROR] Backup directory not found: {backup_dir}")
            return 1
        
        print(f"Restoring database from: {backup_dir}")
        success = registry.restore_from_backup(backup_dir)
        if success:
            print("Database restore completed successfully.")
        else:
            print("Database restore failed.")
            return 1
    
    else:
        print("Unknown registry action. Use --help for available commands.")


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Handle translate command
    if args.command == 'translate':
        success = translate_paper(args.pdf, args.output, args.max_pages, args.academic, getattr(args, 'html_only', False))
        return 0 if success else 1
    
    # Handle registry command
    if args.command == 'registry':
        try:
            handle_registry_command(args)
            return 0
        except Exception as e:
            print(f"[ERROR] Registry command failed: {e}")
            return 1
    
    # Default to search command
    # Print header
    print_header(args.query, args)
    
    # Check Vertex AI (simplified check without actual invocation)
    print("\nChecking API configuration...")
    try:
        from src.core.config import create_llm_model
        # Just create the model to verify configuration, don't invoke it
        model = create_llm_model()
        print(f"[OK] API configured (Vertex AI)")
    except Exception as e:
        print(f"[ERROR] API error: {e}")
        print("\nPlease check your API configuration:")
        print("- For Vertex AI: Set GOOGLE_CLOUD_PROJECT")
        print("- For Google AI: Set GOOGLE_API_KEY")
        return 1
    
    # Run research
    result = run_research(args.query, args)
    
    if result:
        # Handle translation option
        if hasattr(args, 'translate') and args.translate:
            translate_from_results(result, args.translate)
        
        print("\n[SUCCESS] Research completed successfully!")
        return 0
    else:
        print("\n[FAILED] Research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
