#!/usr/bin/env python3
"""
Interactive Q&A Script for AD Documents.

This script allows users to ask questions about Airworthiness Directive
documents that have been processed by the pipeline.

Usage:
    python chat.py                    # Interactive mode
    python chat.py --ingest           # Ingest documents first, then start Q&A
    python chat.py --ask "question"   # Ask single question (non-interactive)
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Interactive Q&A for AD Documents")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents before Q&A")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--ask", "-a", type=str, help="Ask a single question (non-interactive)")
    args = parser.parse_args()
    
    console = Console()
    
    # Initialize pipeline
    config = PipelineConfig(data_dir=args.data_dir)
    pipeline = RAGPipeline(config)
    
    # Check if documents are loaded
    stats = pipeline.get_stats()
    doc_count = stats["vector_store"]["document_count"]
    
    if doc_count == 0 or args.ingest:
        console.print("[yellow]Loading documents...[/yellow]")
        try:
            num_chunks = pipeline.ingest_documents(force_reload=args.ingest)
            console.print(f"[green]Loaded {num_chunks} document chunks[/green]")
        except Exception as e:
            console.print(f"[red]Error loading documents: {e}[/red]")
            return
    
    # Single question mode
    if args.ask:
        console.print(f"\n[bold green]Question:[/bold green] {args.ask}")
        console.print("\n[dim]Searching for answer...[/dim]")
        answer = pipeline.ask(args.ask)
        console.print(Panel(
            Markdown(answer),
            title="Answer",
            border_style="green"
        ))
        return
    
    # Interactive mode
    console.print(Panel.fit(
        "[bold blue]AD Document Q&A Assistant[/bold blue]\n"
        "[dim]Ask anything about Airworthiness Directives[/dim]",
        border_style="blue"
    ))
    
    console.print(f"\n[green]{doc_count} document chunks ready[/green]")
    
    # Show available ADs
    all_docs = pipeline.vector_store.get_all_documents()
    ad_ids = set()
    for doc in all_docs:
        if 'ad_id' in doc.get('metadata', {}):
            ad_ids.add(doc['metadata']['ad_id'])
    
    if ad_ids:
        console.print(f"[cyan]Available ADs: {', '.join(ad_ids)}[/cyan]")
    
    # Instructions
    console.print("\n[dim]Commands:[/dim]")
    console.print("[dim]  - Type your question[/dim]")
    console.print("[dim]  - 'exit' - quit[/dim]")
    console.print("[dim]  - 'stats' - show statistics[/dim]")
    console.print("-" * 50)
    
    # Interactive Q&A loop
    while True:
        try:
            question = console.input("\n[bold green]Question:[/bold green] ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if question.lower() == 'stats':
                stats = pipeline.get_stats()
                console.print(f"\n[cyan]Statistics:[/cyan]")
                console.print(f"   Documents: {stats['vector_store']['document_count']}")
                console.print(f"   ADs: {', '.join(stats['extracted_ads']) or 'Not yet extracted'}")
                continue
            
            console.print("\n[dim]Searching for answer...[/dim]")
            
            answer = pipeline.ask(question)
            
            console.print(Panel(
                Markdown(answer),
                title="Answer",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Session ended.[/yellow]")
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
