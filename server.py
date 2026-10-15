#!/usr/bin/env python3
"""
Cognitive Browser REST API Server entrypoint shim.
Provides full backward compatibility for local execution and legacy container calls.
"""
import sys
import os

# Add root directory to sys.path so cognitive_browser is always importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_browser.server import app, main

if __name__ == "__main__":
    main()
