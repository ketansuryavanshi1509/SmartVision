#!/usr/bin/env python3
"""
Entry point for the SmartVision Professional Package.
Run this script to start the application components.
"""
import sys
import os

# Ensure the root directory is in sys.path for the 'app' package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
