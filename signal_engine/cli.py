import argparse
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import MODEL_MAP
from signal_engine.utils import get_symbols
from signal_engine.runner import run_for_symbol
from src.models.db import SessionLocal, migrate_json_to_db

console = Console()

def _flow(args):
    """Full end-to-end pipeline."""
    console.print("[bold cyan]>>> Starting full pipeline flow...[/bold cyan]")
    
    symbols_to_process = get_symbols(args.syms)
    if not symbols_to_process:
        console.print("[bold red]Error: No symbols found to process. Exiting.[/bold red]")
        return
    
    models_to_run = list(MODEL_MAP.keys()) if args.models and "all" in args.models else args.models

    db = SessionLocal()
    try:
        for symbol in symbols_to_process:
            console.print(f"\n--- Processing Symbol: {symbol} ---")
            for model_key in models_to_run:
                run_for_symbol(symbol, model_key, db)
    finally:
        db.close()
    
    console.print("\n[bold green]✓ Pipeline complete![/bold green]")

def _migrate(args):
    migrate_json_to_db()

def main():
    parser = argparse.ArgumentParser(prog="signal-engine", description="Signal Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    flow_p = subparsers.add_parser("flow", help="Run complete pipeline from fetch to backtest")
    flow_p.add_argument("--syms", nargs="+", required=True, help='Symbols to process (e.g., AAPL MSFT) or "universe".')
    flow_p.add_argument("--models", nargs="+", required=True, help=f'Models to run (e.g., dt xgb) or "all". Available: {", ".join(MODEL_MAP.keys())}')
    
    subparsers.add_parser("migrate", help="Migrate old JSON strategies to DB (legacy)")

    args = parser.parse_args()
    
    command_map = {
        "flow": _flow,
        "migrate": _migrate,
    }
    command_map[args.command](args)

if __name__ == "__main__":
    main()