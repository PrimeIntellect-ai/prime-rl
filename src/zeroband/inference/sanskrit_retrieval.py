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



def parse_prediction(prediction: str) -> dict[str, str]:
  """Parse LLM's prediction into structured fields."""
  result = {}

  # Don't lowercase first - we'll do it selectively
  # This preserves the original for better matching

  # Handle Devanagari numerals
  devanagari_numerals = {
      '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
      '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
  }
  cleaned = prediction
  for dev, eng in devanagari_numerals.items():
      cleaned = cleaned.replace(dev, eng)

  # Split into lines for cleaner parsing
  lines = cleaned.split('\n')
  full_text = ' '.join(lines)

  # More forgiving regex patterns - match until newline, comma, or next field
  # Use (?i) for case-insensitive matching
  genre_match = re.search(r'(?i)genre[:\s]+([^\n,]+?)(?=\s*(?:author|text|chapter|verse|,|\n|$))', full_text)
  author_match = re.search(r'(?i)author[:\s]+([^\n,]+?)(?=\s*(?:text|chapter|verse|,|\n|$))', full_text)
  text_match = re.search(r'(?i)(?:text|work)[:\s]+([^\n,]+?)(?=\s*(?:chapter|verse|,|\n|$))', full_text)
  chapter_match = re.search(r'(?i)(?:chapter|canto|sarga|adhyaya|अध्याय)[:\s]+(\d+)', full_text)
  verse_match = re.search(r'(?i)(?:verse|shloka|sloka|श्लोक)[:\s]+(\d+)', full_text)

  # Extract and clean results
  if genre_match:
      result['genre'] = genre_match.group(1).strip().lower()
  if author_match:
      result['author'] = author_match.group(1).strip().lower()
  if text_match:
      result['text'] = text_match.group(1).strip().lower()
  if chapter_match:
      result['chapter'] = chapter_match.group(1).strip()
  if verse_match:
      result['verse'] = verse_match.group(1).strip()

  # Try natural language patterns if structured format fails
  if not result.get('text') and not result.get('author'):
      # "from Kalidasa's Meghaduta" pattern
      from_match = re.search(r"(?i)from\s+([a-zA-Z\u0900-\u097F]+)'s\s+([a-zA-Z\u0900-\u097F]+)", full_text)
      if from_match:
          result['author'] = from_match.group(1).strip().lower()
          result['text'] = from_match.group(2).strip().lower()

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


def compute_retrieval_reward(prediction: str, ground_truth: Dict[str, str]) -> float:
  """
  Compute reward for Sanskrit literature source retrieval.
  
  Reward Structure (Total: 1.0):
  - Genre: 0.15 (independent)
  - Author: 0.20 (independent) 
  - Text: 0.30 (independent)
  - Chapter: 0.20 (requires correct text)
  - Verse: 0.15 (requires correct text AND chapter)
  
  Examples:
  - All correct: 1.0
  - Wrong genre, correct author/text: 0.50
  - Wrong genre, correct author/text/chapter: 0.70
  - Wrong genre, correct author/text/chapter/verse: 0.85
  - Only text correct: 0.30
  - Text wrong, everything else correct: 0.35 (genre + author only)
  
  Genre Synonyms (partial credit at 0.10):
  - kavya ≈ poetry, poem, mahakavya, epic
  - epic ≈ itihasa, mahakavya, kavya
  - purana ≈ mythology, itihasa
  """


  # Parse the prediction
  try:
      predicted = parse_prediction(prediction)
  except Exception as e:
      return 0.0

  # Normalize fields
  def normalize_field(field: Optional[str]) -> Optional[str]:
      if field is None:
          return None
      return normalize_text_name(field)

  # Individual field rewards
  reward = 0.0

  # Genre (0.15) - independent
  if predicted.get('genre') and ground_truth.get('genre'):
      pred_genre = normalize_field(predicted['genre'])
      true_genre = normalize_field(ground_truth['genre'])

      # Genre synonyms
      genre_synonyms = {
          'kavya': ['poetry', 'poem', 'mahakavya'],
          'epic': ['itihasa', 'mahakavya', 'kavya'],
          'purana': ['mythology', 'itihasa'],
          'veda': ['vedic', 'shruti'],
          'shastra': ['sastra', 'science', 'sutra']
      }

      if pred_genre == true_genre:
          reward += 0.15
      else:
          # Check synonyms
          for main_genre, synonyms in genre_synonyms.items():
              if (true_genre == main_genre and pred_genre in synonyms) or \
                 (pred_genre == main_genre and true_genre in synonyms):
                  reward += 0.10  # Partial credit
                  break

  # Author (0.20) - independent
  if predicted.get('author') and ground_truth.get('author'):
      if normalize_field(predicted['author']) == normalize_field(ground_truth['author']):
          reward += 0.20

  # Text (0.30) - independent
  text_correct = False
  if predicted.get('text') and ground_truth.get('text'):
      if normalize_field(predicted['text']) == normalize_field(ground_truth['text']):
          reward += 0.30
          text_correct = True

  # Chapter (0.20) - depends on text
  chapter_correct = False
  if predicted.get('chapter') and ground_truth.get('chapter'):
      if str(predicted['chapter']) == str(ground_truth['chapter']):
          if text_correct:
              reward += 0.20
              chapter_correct = True
          # No partial credit if text is wrong

  # Verse (0.15) - depends on text AND chapter
  if predicted.get('verse') and ground_truth.get('verse'):
      if str(predicted['verse']) == str(ground_truth['verse']):
          if text_correct and chapter_correct:
              reward += 0.15
          # No partial credit if text or chapter is wrong

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