#!/usr/bin/env python3
"""
Cognitive Browser REST API Server entrypoint shim.
Provides full backward compatibility for local execution and legacy container calls.
"""
import sys
import os
import warnings

warnings.warn(
    "Invoking server.py from the repository root is deprecated. "
    "Please run 'python -m cognitive_browser.server' or the 'cognitive-browser-api' console command.",
    DeprecationWarning,
    stacklevel=2
)

# Add root directory to sys.path so cognitive_browser is always importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_browser.server import app, main

if __name__ == "__main__":
    main()
