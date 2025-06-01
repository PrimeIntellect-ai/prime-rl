# src/zeroband/inference/genesys/sanskrit.py
from typing import Dict, List, Optional, Set # Added Set
import unicodedata

# Corrected and specific chandas imports
from chandas import to_pattern_lines  # Correct: This function is in the top-level chandas package
from chandas.svat.identify.identifier import Identifier as ChandasIdentifier # Import the class
from chandas.svat.data import metrical_data # To access chandas's internal data structures

# --- Global Chandas Setup (Initialize once when this module is loaded) ---
if not metrical_data.all_data:  # Check if already initialized
    metrical_data.InitializeData()

# Create a global instance of the chandas Identifier
# This is efficient as it pre-processes meter data on initialization.
CHANDAS_METER_ENGINE = ChandasIdentifier(metrical_data=metrical_data)
# --- End Global Chandas Setup ---

import unicodedata
import re # Import re for the final cleanup

def normalize_meter_name_for_comparison(meter_name: str) -> str:
    """
    Normalizes a meter name (either user input or from chandas output)
    for consistent comparison. Aims for a simplified ASCII representation.
    """
    if not meter_name:
        return ""
    
    name = meter_name.lower().strip()
    
    # Normalize unicode to NFD (Canonical Decomposition) to separate base characters from diacritics
    # Then remove diacritics (Nonspacing Marks)
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    
    # Specific transliteration simplifications BEFORE general noise removal
    # Order can be important
    name = name.replace('sh', 's') # ś, ṣ -> s (catches common user input)
    name = name.replace('ch', 'c') # No change if already c, but handles user 'ch'
    name = name.replace('ṭ', 't')
    name = name.replace('ḍ', 'd')
    name = name.replace('ṇ', 'n')
    name = name.replace('ṣ', 's') # Redundant if sh->s is done, but safe
    name = name.replace('ś', 's') # Redundant if sh->s is done, but safe
    name = name.replace('ṛ', 'ri') # Common vocalic r representation
    name = name.replace('ṁ', 'm')  # Anusvara
    name = name.replace('ḥ', 'h')  # Visarga
    name = name.replace('ṅ', 'n')
    name = name.replace('ñ', 'n')

    # Handle common IAST long vowels (after diacritics are removed, these won't exist,
    # but good if the input was already simplified)
    name = name.replace('ā', 'a')
    name = name.replace('ī', 'i')
    name = name.replace('ū', 'u')
    
    # Specific known conjuncts from user input (these are less common in canonical names)
    # This part is tricky because chandas names are usually more precise.
    # Focus on making chandas names and simple user names converge.
    # name = name.replace('ṣṭ', 'st') # If you want this
    # name = name.replace('kṣ', 'ksh')
    # name = name.replace('jñ', 'gy')

    # Remove noise characters and common suffixes
    name = name.replace(' ', '').replace('-', '').replace('_', '')
    name = name.replace('(', '').replace(')', '') # Remove parentheses
    name = name.replace('vrtta', '')
    name = name.replace('meter', '')
    name = name.replace('vṛtta', '') # In case NFD didn't fully strip ṛ

    # Optional: Remove any remaining non-alphanumeric characters
    # name = re.sub(r'[^a-z0-9]', '', name) 
    # Be cautious with this, as some meter names might have numbers (though rare in this context)

    return name

def verify_meter_with_chandas(poem_devanagari: str, expected_meter_user_name: str) -> float:
    """
    Verifies the meter of a Devanagari poem against an expected meter using the chandas library.

    Args:
        poem_devanagari: The poem text in Devanagari script.
        expected_meter_user_name: The user-friendly name of the expected meter.

    Returns:
        A score (1.0 for exact match, 0.7 for partial match, 0.0 otherwise).
    """
    try:
        lines = [line.strip() for line in poem_devanagari.strip().split('\n') if line.strip()]
        if not lines:
            # print("Debug (verify): No valid lines in poem.")
            return 0.0

        # Convert Devanagari lines to L/G patterns
        # `to_pattern_lines` is from `chandas/__init__.py`
        input_lg_patterns = to_pattern_lines(lines)
        if not input_lg_patterns or not any(input_lg_patterns):
            # print("Debug (verify): Chandas could not extract L/G patterns.")
            return 0.0

        # Use the global CHANDAS_METER_ENGINE to identify meters
        # This engine is already initialized with metrical_data
        identification_result = CHANDAS_METER_ENGINE.IdentifyFromPatternLines(input_lg_patterns)
        
        # Normalize the user's expected meter name for comparison
        normalized_expected_meter = normalize_meter_name_for_comparison(expected_meter_user_name)
        if not normalized_expected_meter: # If normalization results in empty (e.g. bad input)
            return 0.0

        # print(f"Debug (verify): Normalized Expected: '{normalized_expected_meter}'")
        # print(f"Debug (verify): Chandas ID Result: {identification_result}")

        # Check 'exact' matches (Chandas returns an OrderedSet)
        exact_matches: Optional[Set[str]] = identification_result.get('exact')
        if exact_matches:
            for identified_meter_name in exact_matches:
                if normalize_meter_name_for_comparison(identified_meter_name) == normalized_expected_meter:
                    return 1.0
        
        # Check 'partial' matches if no exact match found
        partial_matches: Optional[Set[str]] = identification_result.get('partial')
        if partial_matches:
            for identified_meter_name in partial_matches:
                if normalize_meter_name_for_comparison(identified_meter_name) == normalized_expected_meter:
                    return 0.7
        
        return 0.0
        
    except Exception as e:
        # It's good practice to log the error or at least print it for debugging during development
        print(f"Chandas error during meter verification: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error info
        return 0.0

def is_repetitive(poem: str, word_uniqueness_threshold=0.3, short_pattern_max_density=0.5) -> bool:
    """
    Checks if the poem exhibits overly repetitive characteristics.
    Args:
        poem: The poem text.
        word_uniqueness_threshold: If unique words / total words is less than this, it's repetitive.
        short_pattern_max_density: If any short pattern (2-5 chars) makes up more than this fraction
                                   of the poem, it's repetitive.
    Returns:
        True if deemed repetitive, False otherwise.
    """
    if not poem.strip():
        return False # Empty poem is not repetitive in this context

    # Normalize poem by removing punctuation and making it lowercase for word analysis
    # For Sanskrit, proper tokenization is complex. Using space as a simple delimiter.
    normalized_poem_for_words = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in poem)
    words = [word for word in normalized_poem_for_words.split() if word]

    if not words: # e.g. poem was only punctuation
        return False
    
    # 1. Check for low word uniqueness
    if len(words) > 3: # Only apply if there are enough words to judge uniqueness
        unique_words = set(words)
        if len(unique_words) / len(words) < word_uniqueness_threshold:
            # print(f"Debug (is_repetitive): Low word uniqueness: {len(unique_words)}/{len(words)}")
            return True
    
    # 2. Check for repeated short character patterns (e.g., "ababab", "lalala")
    #    This helps catch syllable/character-level repetition that word uniqueness might miss.
    #    Analyzing the raw poem string for this.
    poem_no_space = poem.replace(" ", "").replace("\n", "")
    poem_len = len(poem_no_space)
    if poem_len < 6: # Too short to reliably detect char pattern repetition
        return False

    for pattern_len in range(2, min(6, poem_len // 2)): # Check patterns of length 2 to 5
        short_pattern = poem_no_space[:pattern_len]
        # Count occurrences of this specific pattern starting poem
        # More robustly, one could check for any dominant short pattern, not just the initial one.
        # For simplicity, sticking to initial pattern for now.
        # A simple `poem_no_space.count(short_pattern)` could be misleading if patterns overlap heavily.
        # Instead, let's check if the poem seems to be constructed largely of one or two very short patterns.
        
        # This is a more advanced check: if the poem, when compressed (like run-length encoding of sorts),
        # is very small, it might be repetitive.
        # For simplicity, the original `poem.count(pattern) > len(poem) // (i * 2)`
        # was trying to say: if a pattern of length `i` occurs more times than
        # would fill half the poem if patterns were distinct.
        
        # Let's use a slightly modified version of your original logic:
        # Check if the *initial* short pattern repeats excessively
        if poem_no_space.count(short_pattern) * pattern_len > poem_len * short_pattern_max_density:
            # print(f"Debug (is_repetitive): High short pattern density for '{short_pattern}'")
            return True
            
    return False

def compute_sanskrit_poetry_reward(completion: str, verification_info: Dict) -> float:
    """
    Enhanced reward function with anti-gaming measures.
    Assumes 'completion' provides Devanagari text after any <think> tags.
    """
    # 1. Extract poem text
    poem_text_devanagari: str
    if "</think>" in completion:
        parts = completion.split("</think>", 1)
        if len(parts) > 1:
            poem_text_devanagari = parts[1].strip()
        else: # Malformed tag or no content after
            # print("Debug (reward): Malformed <think> tag or no content after.")
            return 0.0
    else:
        # If no <think> tag, assume the whole completion is the poem,
        # but this might be too lenient depending on expected input format.
        # For now, let's be strict: if <think> is expected, it should be there.
        # Or, if the problem implies no <think> tag, then poem_text_devanagari = completion.strip()
        # Assuming for now, if no <think>, it's an invalid format for this specific reward function design.
        # print("Debug (reward): No <think> tag found in completion.")
        return 0.0 # Or handle as per specific requirements for completions without <think>
    
    # 2. Basic content checks
    if not poem_text_devanagari or len(poem_text_devanagari) < 15:  # Adjusted min length
        # print(f"Debug (reward): Poem too short or empty: '{poem_text_devanagari[:10]}...'")
        return 0.0
    
    # 3. Anti-gaming: Repetitiveness check
    if is_repetitive(poem_text_devanagari):
        # print("Debug (reward): Poem deemed repetitive.")
        return 0.0
    
    # 4. Get expected meter from verification info
    #    This will raise KeyError if "meter_type" is not in verification_info,
    #    which is good ("fail fast") if it's a required field.
    try:
        expected_meter_user_name = verification_info["meter_type"]
        # topic = verification_info["topic"] # 'topic' is present but not used in this version
    except KeyError as e:
        print(f"Required field missing in verification_info: {e}")
        return 0.0 # Or handle as a configuration error

    if not expected_meter_user_name:
        # print("Debug (reward): Expected meter name is empty.")
        return 0.0
        
    # 5. Verify meter using the dedicated Chandas function
    meter_score = verify_meter_with_chandas(poem_text_devanagari, expected_meter_user_name)
    
    # Optional: Placeholder for LLM grading for content quality (as in original)
    # content_score, _ = llm_grader.grade_poetry(poem, topic, expected_meter)
    # final_score = (meter_score * 0.7) + (content_score * 0.3) # Example weighting
    
    # For now, returning just the meter score based on chandas verification
    return meter_score