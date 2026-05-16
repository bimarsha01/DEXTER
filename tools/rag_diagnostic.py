"""CLI entry point for RAG diagnostics."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "diagnostics" / "rag_diagnostic.py"
    runpy.run_path(str(target), run_name="__main__")
