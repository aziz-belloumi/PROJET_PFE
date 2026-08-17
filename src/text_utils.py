"""
Utility functions for text formatting and console display.
"""

from __future__ import annotations
import sys
import re

# Range for Arabic Unicode characters (including presentation forms, extensions)
ARABIC_CHAR_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def format_arabic_for_console(text: str) -> str:
    """
    Reshapes and applies BiDi algorithm to Arabic text for proper terminal/console display.
    Connects Arabic letters into proper ligatures and fixes right-to-left visual ordering.
    
    If the text contains no Arabic characters or reshaping fails, returns original text safely.
    """
    if not text or not isinstance(text, str):
        return str(text) if text is not None else ""

    # Only process if text contains Arabic characters
    if not ARABIC_CHAR_PATTERN.search(text):
        return text

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        # Reshape isolated glyphs into contextual cursive forms (initial, medial, final)
        reshaped_text = arabic_reshaper.reshape(text)
        # Apply BiDi layout algorithm to handle visual right-to-left orientation in LTR terminals
        return get_display(reshaped_text)
    except Exception:
        # Fallback gracefully to original text if reshaping fails
        return text


def init_console_encoding() -> None:
    """
    Ensures standard output and error streams handle UTF-8 characters without encoding errors on Windows.
    """
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
