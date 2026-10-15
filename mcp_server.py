#!/usr/bin/env python3
"""
Cognitive Browser MCP Server entrypoint shim.
Provides full backward compatibility for existing MCP client configurations.
"""
import sys
import os
import warnings

warnings.warn(
    "Invoking mcp_server.py from the repository root is deprecated. "
    "Please run 'python -m cognitive_browser.mcp_server' or the 'cognitive-browser' console command.",
    DeprecationWarning,
    stacklevel=2
)

# Add root directory to sys.path so cognitive_browser is always importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_browser.mcp_server import main

if __name__ == "__main__":
    main()
