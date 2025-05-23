# src/zeroband/inference/genesys/sanskrit_utils.py
from typing import Dict, List, Tuple
import re

try:
    from chandas import syllabize, identify
    CHANDAS_AVAILABLE = True
except ImportError:
    CHANDAS_AVAILABLE = False

class SanskritPoetryVerifier:
    """Enhanced verifier using Chandas library."""
    
    def __init__(self):
        if not CHANDAS_AVAILABLE:
            raise ImportError("Chandas library required. Install with: pip install chandas")
        
        # Common Sanskrit meters with their properties
        self.meter_info = {
            "śloka": {
                "syllables_per_pada": 8,
                "padas": 4,
                "description": "Most common Sanskrit meter"
            },
            "anuṣṭubh": {
                "syllables_per_pada": 8,
                "padas": 4,
                "description": "Epic meter, similar to śloka"
            },
            "vasantatilakā": {
                "syllables_per_pada": 14,
                "padas": 4,
                "description": "Spring meter"
            },
            "mandākrāntā": {
                "syllables_per_pada": 17,
                "padas": 4,
                "description": "Slow-stepping meter"
            }
        }
    
    def verify_meter_structure(self, poem: str, expected_meter: str) -> Dict[str, any]:
        """Verify if poem follows expected meter structure."""
        lines = poem.strip().split('\n')
        
        # Get pattern analysis from Chandas
        pattern_lines = identify.to_pattern_lines(lines)
        id_result = identify.identifier.IdentifyFromPatternLines(pattern_lines)
        
        # Syllabize each line
        syllable_counts = []
        for line in lines:
            syllables = syllabize.get_syllables(line)
            syllable_counts.append(len(syllables.split()))
        
        return {
            "identified_meter": id_result.get('exact', 'Unknown'),
            "confidence": self._calculate_confidence(id_result),
            "syllable_counts": syllable_counts,
            "pattern_lines": pattern_lines,
            "matches_expected": self._check_meter_match(id_result, expected_meter)
        }
    
    def _calculate_confidence(self, id_result: Dict) -> float:
        """Calculate confidence score based on Chandas results."""
        if id_result.get('exact'):
            return 1.0
        elif id_result.get('partial'):
            return 0.7
        elif id_result.get('possible'):
            return 0.4
        return 0.0
    
    def _check_meter_match(self, id_result: Dict, expected_meter: str) -> bool:
        """Check if identified meter matches expected."""
        expected_norm = normalize_meter_name(expected_meter)
        
        # Check exact match
        if id_result.get('exact'):
            if normalize_meter_name(id_result['exact']) == expected_norm:
                return True
        
        # Check partial matches
        for match_type in ['partial', 'possible']:
            if id_result.get(match_type):
                for meter in id_result[match_type]:
                    if normalize_meter_name(meter) == expected_norm:
                        return True
        
        return False

def normalize_meter_name(meter_name: str) -> str:
    """Normalize Sanskrit meter names for comparison."""
    # Remove diacritics and normalize
    replacements = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
        'ḷ': 'l', 'ḹ': 'l', 'ṃ': 'm', 'ḥ': 'h',
        'ś': 'sh', 'ṣ': 'sh', 'ñ': 'n', 'ṅ': 'n',
        'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n', 'ḻ': 'l'
    }
    
    result = meter_name.lower().strip()
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result