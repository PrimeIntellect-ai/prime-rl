from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher
import chandas
import unicodedata


class SanskritMeterDatabase:
    """Comprehensive database of Sanskrit meters with their prosodic patterns."""
    
    METERS = {
        # Vedic Meters (Classical Seven)
        'gayatri': {
            'syllables_per_pada': 8,
            'total_syllables': 24,
            'pattern': 'GLGLGLGL' * 3,  # 3 padas of 8 syllables each
            'description': 'the most sacred Vedic meter with 8 syllables per quarter (3×8=24 total)'
        },
        'usnik': {
            'syllables_per_pada': 7,
            'total_syllables': 28,
            'pattern': 'GLGLGLG' * 4,  # 4 padas of 7 syllables each
            'description': 'the Vedic meter with 7 syllables per quarter (4×7=28 total)'
        },
        'anushtubh': {
            'syllables_per_pada': 8,
            'total_syllables': 32,
            'pattern': 'GLGLGLGL' * 4,
            'description': 'the epic meter used in Mahabharata and Ramayana with 8 syllables per quarter'
        },
        'brhati': {
            'syllables_per_pada': 9,
            'total_syllables': 36,
            'pattern': 'GLGLGLGLG' * 4,  # 4 padas of 9 syllables each
            'description': 'the great Vedic meter with 9 syllables per quarter (4×9=36 total)'
        },
        'pankti': {
            'syllables_per_pada': 8,
            'total_syllables': 40,
            'pattern': 'GLGLGLGL' * 5,  # 5 padas of 8 syllables each
            'description': 'the five-fold Vedic meter with 8 syllables per quarter (5×8=40 total)'
        },
        'trishtubh': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'LGLGGLGGGG' * 4,  # Classic trishtubh pattern
            'description': 'the triple-step Vedic meter with 11 syllables per quarter, second most common in Rigveda'
        },
        'jagati': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,  # 4 padas of 12 syllables each
            'description': 'the cosmic Vedic meter with 12 syllables per quarter (4×12=48 total)'
        },
        
        # Classical Śloka
        'sloka': {
            'syllables_per_pada': 8,
            'total_syllables': 32,
            'pattern': 'GLGLGLGL' * 4,
            'description': 'the most common Sanskrit meter with 8 syllables per quarter, identical to Anuṣṭubh'
        },
        
        # Popular 11-syllable Meters
        'indravajra': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'LGLGGLGGGG' * 4,  # ta-ta-ja-ga-ga
            'description': 'the heroic meter with 11 syllables (ta-ta-ja-ga-ga pattern)'
        },
        'upendravajra': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'GLGLGGLGGG' * 4,  # ja-ta-ja-ga-ga
            'description': 'the noble meter with 11 syllables (ja-ta-ja-ga-ga pattern)'
        },
        'upajati': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'LGLGGLGGGGGLGLGGLGGGLGLGGLGGGGGLGLGGLGGG',  # Mixed indravajra/upendravajra
            'description': 'the mixed meter alternating Indravajrā and Upendravajrā'
        },
        'rathoddhata': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'LGLGGLGGGG' * 4,
            'description': 'the chariot-raising meter with 11 syllables per quarter'
        },
        'svagata': {
            'syllables_per_pada': 11,
            'total_syllables': 44,
            'pattern': 'GLGLGGLGGG' * 4,
            'description': 'the welcome meter with 11 syllables per quarter'
        },
        
        # 12-syllable Meters
        'vamsastha': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGG' * 4,  # ja-ta-ja-ra pattern
            'description': 'the bamboo-like meter with 12 syllables (ja-ta-ja-ra pattern)'
        },
        'totaka': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'LLGLLGLLGLLG' * 4,  # Every 3rd syllable heavy
            'description': 'the rapid meter with 12 syllables, every 3rd syllable heavy'
        },
        'harinapluta': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the deer-leap meter with 12 syllables'
        },
        
        # 13-syllable Meters
        'praharshini': {
            'syllables_per_pada': 13,
            'total_syllables': 52,
            'pattern': 'LGLGGLGGGGGGG' * 4,
            'description': 'the delightful meter with 13 syllables per line'
        },
        'atijagati': {
            'syllables_per_pada': 13,
            'total_syllables': 52,
            'pattern': 'GLGLGGLGGGGGG' * 4,
            'description': 'the beyond-cosmic meter with 13 syllables per quarter'
        },
        
        # 14-syllable Meters
        'vasantatilaka': {
            'syllables_per_pada': 14,
            'total_syllables': 56,
            'pattern': 'LGGLLGGLGGLGGG' * 4,  # ta-bha-ja-ja-ga-ga
            'description': 'the spring ornament meter with 14 syllables (ta-bha-ja-ja-ga-ga pattern)'
        },
        'shakkari': {
            'syllables_per_pada': 14,
            'total_syllables': 56,
            'pattern': 'GLGLGGLGGGGGGG' * 4,
            'description': 'the sugar-sweet meter with 14 syllables per quarter'
        },
        
        # 15-syllable Meters
        'malini': {
            'syllables_per_pada': 15,
            'total_syllables': 60,
            'pattern': 'GGGGLGLGGGGLLGG' * 4,  # na-na-ma-ya-ya pattern
            'description': 'the garlanded meter with 15 syllables per line (na-na-ma-ya-ya pattern)'
        },
        'atishakari': {
            'syllables_per_pada': 15,
            'total_syllables': 60,
            'pattern': 'GLGLGGLGGGGGGGG' * 4,
            'description': 'the beyond-sweet meter with 15 syllables per quarter'
        },
        
        # 16-syllable Meters
        'ashti': {
            'syllables_per_pada': 16,
            'total_syllables': 64,
            'pattern': 'GLGLGGLGGGGGGGGG' * 4,
            'description': 'the eight-fold meter with 16 syllables per quarter'
        },
        
        # 17-syllable Meters
        'mandakranta': {
            'syllables_per_pada': 17,
            'total_syllables': 68,
            'pattern': 'GGGLGLGGGGLLGGG' * 4,
            'description': 'the slow-stepping meter with 17 syllables, favored by Kālidāsa'
        },
        'shikharini': {
            'syllables_per_pada': 17,
            'total_syllables': 68,
            'pattern': 'GLGLGGLGGGGGGGGG' * 4,
            'description': 'the peaked meter with 17 syllables per line'
        },
        'atyashti': {
            'syllables_per_pada': 17,
            'total_syllables': 68,
            'pattern': 'GLGLGGLGGGGGGGGGG' * 4,
            'description': 'the beyond-eight meter with 17 syllables per quarter'
        },
        
        # 18-syllable Meters
        'dhriti': {
            'syllables_per_pada': 18,
            'total_syllables': 72,
            'pattern': 'GLGLGGLGGGGGGGGGGG' * 4,
            'description': 'the steadfast meter with 18 syllables per quarter'
        },
        
        # 19-syllable Meters
        'sardullavikridita': {
            'syllables_per_pada': 19,
            'total_syllables': 76,
            'pattern': 'GGGLGLGGGLGGLGGLGGG' * 4,
            'description': 'the tiger\'s play meter with 19 syllables suggesting power and grace'
        },
        'atidhrti': {
            'syllables_per_pada': 19,
            'total_syllables': 76,
            'pattern': 'GLGLGGLGGGGGGGGGGGG' * 4,
            'description': 'the beyond-steadfast meter with 19 syllables per quarter'
        },
        
        # 20-syllable Meters
        'kriti': {
            'syllables_per_pada': 20,
            'total_syllables': 80,
            'pattern': 'GLGLGGLGGGGGGGGGGGGG' * 4,
            'description': 'the accomplished meter with 20 syllables per quarter'
        },
        
        # 21-syllable Meters
        'sragdhara': {
            'syllables_per_pada': 21,
            'total_syllables': 84,
            'pattern': 'GGGLGLGGGLGGLGGLGGGGG' * 4,
            'description': 'the garland-bearing meter with 21 syllables for elaborate compositions'
        },
        'prakriti': {
            'syllables_per_pada': 21,
            'total_syllables': 84,
            'pattern': 'GLGLGGLGGGGGGGGGGGGGG' * 4,
            'description': 'the natural meter with 21 syllables per quarter'
        },
        
        # 22-syllable Meters
        'akriti': {
            'syllables_per_pada': 22,
            'total_syllables': 88,
            'pattern': 'GLGLGGLGGGGGGGGGGGGGGG' * 4,
            'description': 'the formed meter with 22 syllables per quarter'
        },
        
        # 23-syllable Meters
        'vikriti': {
            'syllables_per_pada': 23,
            'total_syllables': 92,
            'pattern': 'GLGLGGLGGGGGGGGGGGGGGGG' * 4,
            'description': 'the transformed meter with 23 syllables per quarter'
        },
        
        # 24-syllable Meters
        'sankriti': {
            'syllables_per_pada': 24,
            'total_syllables': 96,
            'pattern': 'GLGLGGLGGGGGGGGGGGGGGGGG' * 4,
            'description': 'the composed meter with 24 syllables per quarter'
        },
        
        # Special Vedic Meters
        'viraj': {
            'syllables_per_pada': 10,
            'total_syllables': 40,
            'pattern': 'GLGLGLGLGG' * 4,
            'description': 'the shining meter with 10 syllables per quarter'
        },
        'dvipada_viraj': {
            'syllables_per_pada': 10,
            'total_syllables': 20,
            'pattern': 'GLGLGLGLGGGLGLGLGLGG',  # 2×10 syllables
            'description': 'the two-footed shining meter with 2×10 syllables'
        },
        
        # Aesthetic Meters
        'sundara': {
            'syllables_per_pada': 15,
            'total_syllables': 60,
            'pattern': 'GGGGLGLGGGGLLGG' * 4,
            'description': 'the beautiful meter for aesthetic compositions'
        },
        'lalita': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the graceful meter with delicate rhythm'
        },
        'citra': {
            'syllables_per_pada': 16,
            'total_syllables': 64,
            'pattern': 'GLGLGGLGGGGGGGGG' * 4,
            'description': 'the variegated meter with complex patterns'
        },
        'mattebha': {
            'syllables_per_pada': 18,
            'total_syllables': 72,
            'pattern': 'GGGGLGLGGGGGGGGGGG' * 4,
            'description': 'the intoxicated elephant meter with heavy rhythm'
        },
        'campaka': {
            'syllables_per_pada': 14,
            'total_syllables': 56,
            'pattern': 'LGGLLGGLGGLGGG' * 4,
            'description': 'the champak flower meter with fragrant cadence'
        },
        'mallika': {
            'syllables_per_pada': 13,
            'total_syllables': 52,
            'pattern': 'LGLGGLGGGGGGG' * 4,
            'description': 'the jasmine meter with sweet rhythm'
        },
        'shalini': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the modest meter with elegant cadence'
        },
        'panchacamara': {
            'syllables_per_pada': 16,
            'total_syllables': 64,
            'pattern': 'GLGLGGLGGGGGGGGG' * 4,
            'description': 'the five-whisk meter with ornate structure'
        },
        'bhujangaprayata': {
            'syllables_per_pada': 12,
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the serpent\'s motion meter with flowing rhythm'
        },
        
        # Quantitative Meters (Mātrāvṛtta)
        'arya': {
            'syllables_per_pada': 12,  # Approximate, as it's based on mātrās
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the noble quantitative meter based on morae count, common in Prakrit'
        },
        'vaitaliya': {
            'syllables_per_pada': 14,  # Approximate
            'total_syllables': 56,
            'pattern': 'LGGLLGGLGGLGGG' * 4,
            'description': 'the demonic meter derived from measure metres'
        },
        'aparantika': {
            'syllables_per_pada': 12,  # Based on 16 mātrās approximation
            'total_syllables': 48,
            'pattern': 'GLGLGGLGGGGG' * 4,
            'description': 'the western-border meter with sixteen mātrās per quarter'
        }
    }
    
    @classmethod
    def get_meter_info(cls, meter_name: str) -> Optional[Dict]:
        """Get meter information by name (case-insensitive)."""
        normalized = cls._normalize_name(meter_name)
        return cls.METERS.get(normalized)
    
    @classmethod
    def get_available_meters(cls) -> List[str]:
        """Get list of available meter names."""
        return list(cls.METERS.keys())
    
    @classmethod
    def get_meters_by_syllable_count(cls, syllables_per_pada: int) -> List[Dict]:
        """Get all meters with specified syllables per pada."""
        return [
            {**meter, 'name': name} 
            for name, meter in cls.METERS.items() 
            if meter['syllables_per_pada'] == syllables_per_pada
        ]
    
    @classmethod
    def get_vedic_meters(cls) -> List[str]:
        """Get list of classical Vedic meters."""
        vedic_meters = ['gayatri', 'usnik', 'anushtubh', 'brhati', 'pankti', 'trishtubh', 'jagati', 'viraj']
        return [name for name in vedic_meters if name in cls.METERS]
    
    @classmethod
    def get_classical_meters(cls) -> List[str]:
        """Get list of popular classical meters."""
        classical_meters = ['sloka', 'indravajra', 'upendravajra', 'vasantatilaka', 'mandakranta', 'sardullavikridita', 'sragdhara']
        return [name for name in classical_meters if name in cls.METERS]

    @classmethod
    def _normalize_name(cls, meter_name: str) -> str:
        """Normalize meter name for consistent matching."""
        # Handle common variations and transliterations
        name = meter_name.lower().strip()
        
        # Use NFC to ensure characters are in composed form for proper replacement
        name = unicodedata.normalize('NFC', name)
        
        # Handle conjunct consonants first (before individual character replacements)
        name = name.replace('ṣṭ', 'sht')  # Common conjunct: ṣṭ → sht
        name = name.replace('kṣ', 'ksh')  # kṣ → ksh
        name = name.replace('jñ', 'gn')   # jñ → gn
        
        # Handle compound word patterns (Sanskrit-specific)
        name = name.replace('ūla', 'ulla')  # Śārdūla → Śārdulla (tiger compounds)
        name = name.replace('ūl', 'ull')    # General ūl → ull pattern
        
        # Comprehensive diacritical character mappings
        diacritical_map = {
            # Vowels with macrons
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
            'Ā': 'a', 'Ī': 'i', 'Ū': 'u', 'Ē': 'e', 'Ō': 'o',
            
            # Retroflex consonants  
            'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n', 'ṣ': 's', 'ḷ': 'l',
            'Ṭ': 't', 'Ḍ': 'd', 'Ṇ': 'n', 'Ṣ': 's', 'Ḷ': 'l',
            
            # Sibilants
            'ś': 's', 'Ś': 's',
            
            # Vowels with other marks
            'ṛ': 'r', 'ṝ': 'r', 'ḹ': 'l',
            'Ṛ': 'r', 'Ṝ': 'r', 'Ḹ': 'l',
            
            # Nasals
            'ṅ': 'n', 'ñ': 'n', 'ṃ': 'm',
            'Ṅ': 'n', 'Ñ': 'n', 'Ṃ': 'm',
            
            # Aspirate
            'ḥ': 'h', 'Ḥ': 'h'
        }
        
        # Apply all diacritical mappings
        for diac_char, simple_char in diacritical_map.items():
            name = name.replace(diac_char, simple_char)
        
        # Remove spaces, hyphens, and common variations
        name = name.replace(' ', '').replace('-', '').replace('_', '')
        name = name.replace('vrtta', '').replace('vṛtta', '')
        
        return name

class SanskritProsodyAnalyzer:
    """Analyzes Sanskrit text for prosodic patterns (L/G sequences)."""
    
    def extract_prosodic_patterns(self, text_lines: List[str]) -> List[str]:
        """
        Extract L/G prosodic patterns from Sanskrit text lines.
        
        Args:
            text_lines: List of Sanskrit text lines
            
        Returns:
            List of L/G pattern strings (one per line)
        """
        return chandas.to_pattern_lines(text_lines)


class MetricalSimilarityCalculator:
    """Calculates similarity between prosodic patterns."""
    
    @staticmethod
    def calculate_similarity(pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two L/G patterns.
        
        Returns:
            Float between 0.0 (no similarity) and 1.0 (identical)
        """
        if not pattern1 or not pattern2:
            return 0.0
        
        # Use SequenceMatcher for robust similarity calculation
        matcher = SequenceMatcher(None, pattern1, pattern2)
        return matcher.ratio()
    
    @staticmethod
    def find_best_match_score(input_pattern: str, target_patterns: List[str]) -> float:
        """Find best similarity score against multiple target patterns."""
        if not input_pattern or not target_patterns:
            return 0.0
        
        best_score = 0.0
        for target in target_patterns:
            score = MetricalSimilarityCalculator.calculate_similarity(input_pattern, target)
            best_score = max(best_score, score)
        
        return best_score


def compute_sanskrit_poetry_reward(completion: str, verification_info: Dict) -> float:
    """
    Main reward function for Sanskrit poetry evaluation.
    
    Args:
        completion: Model completion containing Sanskrit poetry
        verification_info: Dict with 'meter_type' and 'topic' keys
        
    Returns:
        Float reward between 0.0 and 1.0
    """
    try:
        # Extract poem from completion
        poem_text = _extract_poem_from_completion(completion)
        if not poem_text:
            return 0.0
        
        # Basic quality checks
        if len(poem_text) < 20 or _is_repetitive_text(poem_text):
            return 0.0
        
        # Get expected meter information
        meter_name = verification_info.get('meter_type', '')
        meter_info = SanskritMeterDatabase.get_meter_info(meter_name)
        
        if not meter_info:
            # Unknown meter - fail completely
            return 0.0
        
        # Analyze prosodic patterns using chandas library
        analyzer = SanskritProsodyAnalyzer()
        lines = poem_text.strip().split('\n')
        extracted_patterns = analyzer.extract_prosodic_patterns(lines)
        
        # If no patterns are extracted, return 0
        if not extracted_patterns:
            return 0.0
        
        # Compare against expected meter pattern
        full_input_pattern = ''.join(extracted_patterns)
        expected_pattern = meter_info['pattern']
        
        # Calculate similarity
        calculator = MetricalSimilarityCalculator()
        similarity = calculator.calculate_similarity(full_input_pattern, expected_pattern)
        
        # Apply bonuses for excellent matches
        if similarity > 0.95:
            similarity = min(1.0, similarity + 0.05)
        
        return similarity
    
    except Exception as e:
        # If any exception occurs (likely from chandas), return 0
        print(f"Error in computing Sanskrit poetry reward: {e}")
        return 0.0


def _extract_poem_from_completion(completion: str) -> str:
    """Extract poem text from model completion."""
    if "</think>" in completion:
        return completion.split("</think>")[1].strip()
    return completion.strip()


def _is_repetitive_text(text: str) -> bool:
    """Check if text is repetitive (anti-gaming measure)."""
    words = text.split()
    if len(words) < 4:
        return False
    
    # Check for repeated words/phrases
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # If any word appears more than 50% of the time, consider repetitive
    max_frequency = max(word_counts.values()) / len(words)
    return max_frequency > 0.5
