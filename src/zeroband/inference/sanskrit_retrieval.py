"""Sanskrit Literature Source Retrieval Reward Function for prime-rl."""

import re
import json
from typing import Dict, Optional, Tuple, Literal
import logging

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# Genre mapping based on GRETIL directory structure
GENRE_MAPPING = {
    "1_veda": "veda",
    "2_epic": "epic", 
    "3_purana": "purana",
    "4_rellit": "religious",
    "5_poetry": "poetry",
    "6_sastra": "shastra",
    "7_fromindonesia": "indonesian"
}


def parse_prediction(prediction: str) -> Dict[str, str]:
    """Parse LLM's prediction into structured fields.
    
    Expected formats (flexible):
    - "Genre: kavya, Author: kalidasa, Text: meghaduta, Chapter: 1, Verse: 15"
    - "This is from Kalidasa's Meghaduta, verse 1.15"
    - "मेघदूत, कालिदास, श्लोक १५"
    """
    result = {}
    
    # Normalize the prediction
    pred_lower = prediction.lower()
    
    # Handle Devanagari numerals
    devanagari_numerals = {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    }
    for dev, eng in devanagari_numerals.items():
        pred_lower = pred_lower.replace(dev, eng)
    
    # Try structured format first
    genre_match = re.search(r'genre[:\s]+([a-zA-Z\u0900-\u097F]+)', pred_lower)
    author_match = re.search(r'author[:\s]+([a-zA-Z\u0900-\u097F\s]+?)(?:,|$)', pred_lower)
    text_match = re.search(r'(?:text|work)[:\s]+([a-zA-Z\u0900-\u097F\s]+?)(?:,|$)', pred_lower)
    chapter_match = re.search(r'(?:chapter|canto|sarga|adhyaya|अध्याय)[:\s]+(\d+)', pred_lower)
    verse_match = re.search(r'(?:verse|shloka|sloka|श्लोक)[:\s]+(\d+)', pred_lower)
    
    # Try natural language patterns
    if not text_match:
        # "from Kalidasa's Meghaduta"
        from_match = re.search(r"from\s+([a-zA-Z\u0900-\u097F]+)'s\s+([a-zA-Z\u0900-\u097F]+)", pred_lower)
        if from_match:
            author_match = from_match.group(1)
            text_match = from_match.group(2)
            result['author'] = author_match.strip()
            result['text'] = text_match.strip()
    
    # Extract what we found
    if genre_match:
        result['genre'] = genre_match.group(1).strip()
    if author_match and 'author' not in result:
        result['author'] = author_match.group(1).strip()
    if text_match and 'text' not in result:
        result['text'] = text_match.group(1).strip()
    if chapter_match:
        result['chapter'] = chapter_match.group(1)
    if verse_match:
        result['verse'] = verse_match.group(1)
    
    return result


Script = Literal["devanagari", "iast", "slp1", "hk"]


def to_devanagari(text: str, script: Script = "iast") -> str:
    """Convert text to Devanagari script.
    
    Args:
        text: Input text
        script: Source script (devanagari, iast, slp1, hk)
        
    Returns:
        Text in Devanagari script
    """
    if not TRANSLITERATION_AVAILABLE:
        return text
        
    if script == "devanagari":
        return text
        
    script_map = {
        "iast": sanscript.IAST,
        "slp1": sanscript.SLP1, 
        "hk": sanscript.HK
    }
    
    if script in script_map:
        return transliterate(text, script_map[script], sanscript.DEVANAGARI)
    
    return text


def to_iast(text: str, script: Script = "devanagari") -> str:
    """Convert text to IAST (International Alphabet of Sanskrit Transliteration).
    
    Args:
        text: Input text
        script: Source script
        
    Returns:
        Text in IAST
    """
    if not TRANSLITERATION_AVAILABLE:
        return text
        
    if script == "iast":
        return text
        
    script_map = {
        "devanagari": sanscript.DEVANAGARI,
        "slp1": sanscript.SLP1,
        "hk": sanscript.HK
    }
    
    if script in script_map:
        return transliterate(text, script_map[script], sanscript.IAST)
    
    return text


def normalize_text_name(text: str) -> str:
    """Normalize text names for comparison."""
    # Remove diacritics and normalize
    replacements = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
        'ḷ': 'l', 'ḹ': 'l', 'ṃ': 'm', 'ḥ': 'h',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ś': 's', 'ṣ': 's'
    }
    
    normalized = text.lower()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # Remove common suffixes
    normalized = re.sub(r'(sutra|sastra|purana|samhita)$', '', normalized)
    
    return normalized.strip()


def is_gaming_attempt(text: str) -> bool:
    """Check if text is a gaming attempt (repetitive patterns)."""
    # Remove whitespace and punctuation
    cleaned = re.sub(r'[\s\W]+', '', text)
    
    if len(cleaned) < 10:
        return False
    
    # Check for single character repetition
    char_counts = {}
    for char in cleaned:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    most_common_ratio = max(char_counts.values()) / len(cleaned)
    if most_common_ratio > 0.7:  # More than 70% same character
        return True
    
    # Check for repeated short patterns
    words = text.split()
    if len(words) >= 4:
        unique_words = len(set(words))
        if unique_words < len(words) * 0.3:  # Less than 30% unique
            return True
    
    return False


def compute_retrieval_reward(prediction: str, ground_truth: Dict[str, str]) -> float:
    """
    Compute hierarchical reward for Sanskrit literature source retrieval.
    
    Args:
        prediction: LLM's prediction as a string
        ground_truth: Dictionary with keys: genre, author, text, chapter, verse
        
    Returns:
        float: Reward between 0 and 1
        
    Scoring:
        - Genre: 0.1 (foundational)
        - Author: 0.15 (requires genre)
        - Text: 0.25 (requires author) 
        - Chapter: 0.25 (requires text)
        - Verse: 0.25 (requires chapter)
    """
    reward = 0.0
    
    # Check for gaming attempts first
    if is_gaming_attempt(prediction):
        logger.warning("Gaming attempt detected - repetitive pattern")
        return 0.0
    
    # Parse the prediction
    try:
        predicted = parse_prediction(prediction)
    except Exception as e:
        logger.warning(f"Failed to parse prediction: {e}")
        return 0.0
    
    # Normalize for comparison (handle missing diacritics)
    def normalize_field(field: Optional[str]) -> Optional[str]:
        if field is None:
            return None
        # Use our normalize_text_name function which removes diacritics
        return normalize_text_name(field)
    
    # Check genre (0.1)
    if predicted.get('genre') and ground_truth.get('genre'):
        pred_genre = normalize_field(predicted['genre'])
        true_genre = normalize_field(ground_truth['genre'])
        
        # Allow for genre synonyms
        genre_synonyms = {
            'kavya': ['poetry', 'poem'],
            'epic': ['itihasa', 'mahakavya'],
            'purana': ['mythology'],
            'veda': ['vedic', 'shruti'],
            'shastra': ['sastra', 'science']
        }
        
        if pred_genre == true_genre:
            reward += 0.1
        else:
            # Check synonyms
            for main_genre, synonyms in genre_synonyms.items():
                if true_genre == main_genre and pred_genre in synonyms:
                    reward += 0.1
                    break
    
    # Only continue if genre is correct
    if reward >= 0.1:
        # Check author (0.15)
        if predicted.get('author') and ground_truth.get('author'):
            pred_author = normalize_text_name(predicted['author'])
            true_author = normalize_text_name(ground_truth['author'])
            
            if pred_author == true_author:
                reward += 0.15
                
                # Check text (0.25)
                if predicted.get('text') and ground_truth.get('text'):
                    pred_text = normalize_text_name(predicted['text'])
                    true_text = normalize_text_name(ground_truth['text'])
                    
                    if pred_text == true_text:
                        reward += 0.25
                        
                        # Check chapter (0.25)
                        if predicted.get('chapter') and ground_truth.get('chapter'):
                            if str(predicted['chapter']) == str(ground_truth['chapter']):
                                reward += 0.25
                                
                                # Check verse (0.25)
                                if predicted.get('verse') and ground_truth.get('verse'):
                                    if str(predicted['verse']) == str(ground_truth['verse']):
                                        reward += 0.25
    
    return reward


def has_sanskrit_content(text: str) -> bool:
    """Check if text contains Sanskrit content (Devanagari or reasonable IAST)."""
    # Check for Devanagari
    if re.search(r'[\u0900-\u097F]', text):
        return True
    
    # Check for IAST patterns (Sanskrit-like words)
    # Look for typical Sanskrit word patterns
    sanskrit_patterns = [
        r'\b[a-zA-Z]*a[mh]\b',  # Words ending in 'am' or 'ah'
        r'\b[a-zA-Z]*sy[a]\b',  # Genitive endings
        r'\b[a-zA-Z]*t[aeiu]\b',  # Common verb endings
        r'\bdharma|karma|yoga|veda|purana|shastra|kavya\b',  # Common terms
    ]
    
    for pattern in sanskrit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False


def format_quote_prompt(quote: str, metadata: Dict[str, str]) -> str:
    """Format a quote and its metadata into a prompt for the LLM."""
    prompt = f"""Identify the source of this Sanskrit quote:

"{quote}"

Please provide:
- Genre (e.g., kavya, epic, purana, veda, shastra)
- Author
- Text/Work name
- Chapter/Canto number (if applicable)
- Verse number

Answer in a structured format."""
    
    return prompt