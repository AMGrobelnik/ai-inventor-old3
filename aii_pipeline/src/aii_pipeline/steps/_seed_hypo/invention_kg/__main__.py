#!/usr/bin/env python3
"""
Entry point for running the invention_kg pipeline as a module.

Usage:
    python -m aii_pipeline.steps._seed_hypo
"""

from .pipeline import main

if __name__ == "__main__":
    main()
