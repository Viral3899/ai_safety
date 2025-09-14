#!/usr/bin/env python3
"""
Simple demo runner for AI Safety Models POC.

This script bypasses package installation issues by directly importing from the src directory.
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import and run the CLI demo
if __name__ == "__main__":
    try:
        from demo.cli_demo import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying alternative import method...")
        
        # Alternative approach - run the demo directly
        os.chdir(os.path.dirname(__file__))
        exec(open('demo/cli_demo.py').read())
