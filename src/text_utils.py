#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for console display and UTF-8 encoding.
"""

import os
import sys


def init_console_encoding() -> None:
    """
    Ensures standard input, output and error streams handle UTF-8 characters without encoding errors on Windows.
    """
    if os.name == "nt":
        try:
            os.system("chcp 65001 > nul 2>&1")
        except Exception:
            pass

    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        if sys.stdin and hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
