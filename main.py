#!/usr/bin/env python3
"""
Main script for AD Extraction Pipeline using RAG.

This script demonstrates the complete workflow:
1. Load PDFs from ./data folder
2. Chunk documents
3. Generate embeddings using NVIDIA API
4. Store in ChromaDB vector database
5. Extract rules using LLM
6. Evaluate test aircraft configurations

Usage:
    python main.py
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, PipelineConfig
from src.models import ADExtractedOutput


def create_test_aircraft() -> list:
    """Create test aircraft configurations from the assignment."""
    test_cases = [
        # Required test cases
        ("MD-11", 48123, []),
        ("DC-10-30F", 47890, []),
        ("Boeing 737-800", 30123, []),
        ("A320-214", 5234, []),
        ("A320-232", 6789, ["mod 24591 (production)"]),
        ("A320-214", 7456, ["SB A320-57-1089 Rev 04"]),
        ("A321-111", 8123, []),
        ("A321-112", 364, ["mod 24977 (production)"]),
        ("A319-100", 9234, []),
        ("MD-10-10F", 46234, []),
        
        # Verification cases
        ("MD-11F", 48400, []),
        ("A320-214", 4500, ["mod 24591 (production)"]),
        ("A320-214", 4500, []),
    ]
    
    return test_cases


def main():
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Airworthiness Directive Extraction Pipeline[/bold blue]\n"
        "[dim]Using RAG with NVIDIA Embeddings + ChromaDB + LLM[/dim]",
        border_style="blue"
    ))
    
    # Initialize pipeline
    data_dir = Path(__file__).parent / "data"
    config = PipelineConfig(data_dir=str(data_dir))
    pipeline = RAGPipeline(config)
    
    # =========================================
    # STEP 1 & 2: Load and Chunk PDFs, then Embed and Store
    # =========================================
    console.print("\n[bold cyan]PHASE 1: Document Ingestion[/bold cyan]")
    console.print("-" * 50)
    
    try:
        num_chunks = pipeline.ingest_documents(force_reload=True)
        console.print(f"[green]Stored {num_chunks} document chunks in ChromaDB[/green]")
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        console.print("[yellow]Continuing with existing data if available...[/yellow]")
    
    # Show vector store stats
    stats = pipeline.get_stats()
    console.print(f"   Vector store: {stats['vector_store']['document_count']} documents")
    
    # =========================================
    # STEP 3: Extract Rules using LLM
    # =========================================
    console.print("\n[bold cyan]PHASE 2: Rule Extraction with LLM[/bold cyan]")
    console.print("-" * 50)
    
    try:
        extracted_rules = pipeline.extract_all_rules()
        
        for ad_id, rules in extracted_rules.items():
            console.print(f"\n[green]{ad_id}[/green]")
            console.print(f"   Models: {', '.join(rules.applicability_rules.aircraft_models[:5])}...")
            console.print(f"   Exclusions: {rules.applicability_rules.excluded_if_modifications}")
    except Exception as e:
        console.print(f"[red]Error during extraction: {e}[/red]")
        raise
    
    # =========================================
    # STEP 4: Export Structured Rules
    # =========================================
    console.print("\n[bold cyan]PHASE 3: Export Structured Rules[/bold cyan]")
    console.print("-" * 50)
    
    output_path = pipeline.export_rules("output/extracted_rules.json")
    console.print(f"Rules exported to: {output_path}")
    
    # Print example JSON output
    console.print("\n[bold]Sample JSON Output:[/bold]")
    for ad_id, rules in list(extracted_rules.items())[:1]:
        console.print(json.dumps(rules.model_dump(), indent=2))
    
    # =========================================
    # STEP 5: Evaluate Test Aircraft
    # =========================================
    console.print("\n[bold cyan]PHASE 4: Aircraft Evaluation[/bold cyan]")
    console.print("-" * 50)
    
    test_aircraft = create_test_aircraft()
    
    # Create results table
    table = Table(title="Aircraft Evaluation Results")
    table.add_column("Aircraft Model", style="cyan")
    table.add_column("MSN", style="magenta")
    table.add_column("Modifications", style="yellow")
    table.add_column("FAA AD 2025-23-53", style="green")
    table.add_column("EASA AD 2025-0254", style="blue")
    
    evaluation_results = []
    
    for model, msn, mods in test_aircraft:
        results = pipeline.evaluate_aircraft(model, msn, mods)
        
        faa_result = None
        easa_result = None
        
        for result in results:
            if "FAA" in result.ad_id:
                faa_result = result
            elif "EASA" in result.ad_id:
                easa_result = result
        
        faa_status = "Affected" if faa_result and faa_result.is_affected else "Not affected"
        easa_status = "Affected" if easa_result and easa_result.is_affected else "Not affected"
        
        mods_str = ", ".join(mods) if mods else "None"
        
        table.add_row(
            model,
            str(msn),
            mods_str[:30],
            faa_status,
            easa_status
        )
        
        evaluation_results.append({
            "aircraft_model": model,
            "msn": msn,
            "modifications": mods,
            "faa_ad_2025_23_53": {
                "affected": faa_result.is_affected if faa_result else False,
                "reason": faa_result.reason if faa_result else "AD not found"
            },
            "easa_ad_2025_0254": {
                "affected": easa_result.is_affected if easa_result else False,
                "reason": easa_result.reason if easa_result else "AD not found"
            }
        })
    
    console.print(table)
    
    # Export evaluation results
    with open("output/evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    console.print("\nEvaluation results exported to: output/evaluation_results.json")
    
    # =========================================
    # Verification
    # =========================================
    console.print("\n[bold cyan]PHASE 5: Verification[/bold cyan]")
    console.print("-" * 50)
    
    verification = Table(title="Verification Against Expected Results")
    verification.add_column("Aircraft", style="cyan")
    verification.add_column("Expected FAA", style="green")
    verification.add_column("Actual FAA", style="green")
    verification.add_column("Expected EASA", style="blue")
    verification.add_column("Actual EASA", style="blue")
    verification.add_column("Status", style="bold")
    
    verifications = [
        ("MD-11F MSN 48400", True, False),
        ("A320-214 MSN 4500 w/mod 24591", False, False),
        ("A320-214 MSN 4500 no mods", False, True),
    ]
    
    for i, (desc, exp_faa, exp_easa) in enumerate(verifications):
        actual = evaluation_results[-(3-i)]
        act_faa = actual["faa_ad_2025_23_53"]["affected"]
        act_easa = actual["easa_ad_2025_0254"]["affected"]
        
        faa_match = "Pass" if act_faa == exp_faa else "Fail"
        easa_match = "Pass" if act_easa == exp_easa else "Fail"
        
        verification.add_row(
            desc,
            "Affected" if exp_faa else "Not affected",
            "Affected" if act_faa else "Not affected",
            "Affected" if exp_easa else "Not affected",
            "Affected" if act_easa else "Not affected",
            f"{faa_match} {easa_match}"
        )
    
    console.print(verification)
    
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")


if __name__ == "__main__":
    main()
