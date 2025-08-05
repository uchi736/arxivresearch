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
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.core.workflow import build_advanced_workflow
from src.core.models import AdvancedAgentState
from src.core.progress_tracker import ProgressTracker, StepStatus


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
        """
    )
    
    parser.add_argument("query", help="Search query (Japanese or English)")
    parser.add_argument("--depth", choices=["shallow", "moderate", "deep"], 
                       default="moderate", help="Analysis depth")
    parser.add_argument("--workflow", choices=["standard"], 
                       default="standard", help="Workflow type")
    parser.add_argument("--papers", type=int, default=5, 
                       help="Number of papers to analyze")
    parser.add_argument("--output", help="Output file for report (markdown)")
    parser.add_argument("--token-budget", type=int, default=30000,
                       help="Token budget for analysis")
    
    return parser.parse_args()


def build_workflow(workflow_type: str):
    """Build the specified workflow"""
    if workflow_type != "standard":
        raise ValueError("Only 'standard' workflow is supported.")
    
    workflow_names = {
        "standard": "Standard (with RAG)"
    }
    
    print(f"\nBuilding workflow: {workflow_names[workflow_type]}")
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
    """Run the research workflow"""
    progress = CLIProgressDisplay()
    
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
            "progress_tracker": None
        }
        
        # Override paper count based on depth
        if args.depth == "shallow":
            args.papers = min(args.papers, 10)
        elif args.depth == "moderate":
            args.papers = min(args.papers, 5)
        else:  # deep
            args.papers = min(args.papers, 3)
        
        print(f"\nStarting research...")
        start_time = time.time()
        
        # Progress updates
        progress.update("Planning research...", 0.1)
        
        # Execute workflow
        result = workflow.invoke(initial_state)
        
        elapsed = time.time() - start_time
        progress.complete(f"Done in {elapsed:.1f}s")
        
        # Display results
        display_results(result, args.output)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n[!] Research interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n\n[ERROR] {type(e).__name__}: {e}")
        logger.exception("Research failed")
        return False


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print_header(args.query, args)
    
    # Check Vertex AI
    print("\nChecking API configuration...")
    try:
        from src.core.config import create_llm_model
        model = create_llm_model()
        response = model.invoke("Say OK")
        print(f"[OK] API connected (Vertex AI)")
    except Exception as e:
        print(f"[ERROR] API error: {e}")
        print("\nPlease check your API configuration:")
        print("- For Vertex AI: Set GOOGLE_CLOUD_PROJECT")
        print("- For Google AI: Set GOOGLE_API_KEY")
        return 1
    
    # Run research
    success = run_research(args.query, args)
    
    if success:
        print("\n[SUCCESS] Research completed successfully!")
        return 0
    else:
        print("\n[FAILED] Research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
