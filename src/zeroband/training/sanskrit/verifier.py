"""Sanskrit meter verification utilities."""
from __future__ import annotations
from typing import Literal, Optional, Dict, List

import chandas
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

Script = Literal["devanagari", "iast", "slp1", "hk"]

def _to_devanagari(text: str, script: Script) -> str:
    """
    Convert text to Devanagari script.
    
    Args:
        text: Input text in any supported script
        script: Source script name
        
    Returns:
        Text converted to Devanagari
    """
    if script == "devanagari":
        return text
    mapping = {
        "iast": sanscript.IAST,
        "slp1": sanscript.SLP1,
        "hk": sanscript.HK,
    }[script]
    return transliterate(text, mapping, sanscript.DEVANAGARI)

def verify_meter(
    poem: str, 
    expected_meter: str, 
    script: Script = "devanagari", 
    strict_match: bool = True
) -> Dict[str, List[str]]:
    """
    Verify if a poem matches the expected meter.
    
    Args:
        poem: Input poem text
        expected_meter: Name of the expected meter
        script: Script of the input text
        strict_match: If True, only exact matches count. If False, accidental matches also count.
        
    Returns:
        Dictionary with keys:
        - exact: List of exactly matching meters
        - accidental: List of accidentally matching meters
    """
    poem_deva = _to_devanagari(poem, script)
    lines = [ln.strip() for ln in poem_deva.splitlines() if ln.strip()]
    patterns = chandas.to_pattern_lines(lines)
    result = chandas.svat_identifier.IdentifyFromPatternLines(patterns)
    
    # Get the actual values
    exact = list(result.get("exact", []))
    accidental = list(result.get("accidental", []))
    
    return {
        "exact": exact,
        "accidental": accidental,
        "matches_expected": (
            expected_meter in exact if strict_match 
            else expected_meter in exact + accidental
        )
    }
