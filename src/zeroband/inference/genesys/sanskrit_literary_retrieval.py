import json
import re
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz # pip install fuzzywuzzy python-Levenshtein

VERBOSE_DEBUG_SCORING = False # Set to True for detailed scoring logs

SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER = ["WorkTitle", "Genre", "Book", "Chapter", "Verse"]

def extract_predicted_source_of_origin(model_response: str) -> str:
    """
    Extract the predicted source of origin from the model response.
    """
    pass


def normalize_string_for_comparison(text: str) -> str:
    """
    Normalizes strings for robust comparison:
    - Lowercase
    - Basic Latin character equivalents for common Sanskrit diacritics
    - Normalize multiple whitespaces to a single space
    - Strip leading/trailing common punctuation that might interfere with matching.
    """
    if text is None:
        return None
    text_str = str(text).lower().strip()
    
    # Character-level diacritic simplification
    # copied from https://github.com/PrimeIntellect-ai/prime-rl/pull/302
    diacritic_replacements = {
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
        'ḥ': 'h', 'Ḥ': 'h',
    }
    for original, replacement in diacritic_replacements.items():
        text_str = text_str.replace(original, replacement)
    
    # Normalize common punctuation and spacing
    text_str = re.sub(r'\s*-\s*', '-', text_str) # Normalize hyphens if used in GT, e.g. "(mitra-ed.)"
    text_str = re.sub(r'\s*\(.*?\)\s*$', '', text_str) # Optionally remove all parenthetical suffixes like (accented), (madhyandina) for broader matching
    # If you want to keep parentheticals for stricter matching, comment out the line above.
    
    text_str = re.sub(r'\s+', ' ', text_str).strip() # Normalize internal and external whitespace
    # Remove leading/trailing punctuation that isn't part of core ID, but keep internal dots/hyphens.
    text_str = re.sub(r"^[.:,\-\s\'\"]+", "", text_str)
    text_str = re.sub(r"[.:,\-\s\'\"]+$", "", text_str)
    return text_str if text_str else None


def parse_llm_answer_to_universal(llm_answer_string: str) -> dict:
    """
    Parses the LLM's answer (expected in "Label: Value" format within <answer> tags)
    into a dictionary with universal keys and normalized values.
    """
    pred_universal = {label: None for label in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER}
    if not llm_answer_string:
        return pred_universal

    answer_match = re.search(r"<\s*answer\s*>(.*?)</\s*answer\s*>", llm_answer_string, re.DOTALL | re.IGNORECASE)
    content_to_parse = llm_answer_string 
    if answer_match:
        content_to_parse = answer_match.group(1).strip()
    else:
        if VERBOSE_DEBUG_SCORING: print(f"LLM_PARSE_DEBUG: <answer> tags not found in: '{llm_answer_string[:100]}...'")

    lines = content_to_parse.split('\n')
    for line in lines:
        line = line.strip()
        for label in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER:
            match = re.match(rf"^\s*{label}\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                if value_str.lower() in ["unknown", "n/a", "none", ""]:
                    pred_universal[label] = None
                else:
                    pred_universal[label] = normalize_string_for_comparison(value_str)
                break 
    return pred_universal


def llm_scoring(pred_universal_dict: dict, gt_universal_dict: dict) -> float:
    """
    Calculates a hierarchical score by comparing predicted universal components
    against ground truth universal components.
    Ground truth is assumed to already have the universal fields with descriptive, normalized values.
    """
    if not gt_universal_dict: 
        if VERBOSE_DEBUG_SCORING: print("SCORE_DEBUG: Ground truth universal dict is empty or None.")
        is_pred_empty_or_unknown = all(
            (pred_universal_dict.get(lbl) is None or pred_universal_dict.get(lbl).lower() == "unknown")
            for lbl in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER
        )
        return 1.0 if is_pred_empty_or_unknown else 0.0

    weights = {'WorkTitle': 0.40, 'Genre': 0.10, 'Book': 0.20, 'Chapter': 0.15, 'Verse': 0.15}
    achieved_score = 0.0
    
    # Calculate max possible score for this specific GT sample
    max_possible_score_for_sample = 0.0
    for label in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER:
        if gt_universal_dict.get(label) is not None:
            max_possible_score_for_sample += weights[label]
    
    work_title_matched_sufficiently = False

    # score WorkTitle
    gt_work_title_val = gt_universal_dict.get('WorkTitle')
    pred_work_title_val = pred_universal_dict.get('WorkTitle')
    if gt_work_title_val and pred_work_title_val: # Both must exist to compare
        ratio = fuzz.token_set_ratio(pred_work_title_val, gt_work_title_val) 
        
        if ratio >= 85: # high confidence match threshold
            achieved_score += weights['WorkTitle'] * (ratio / 100.0)
            work_title_matched_sufficiently = True # to allow sub-levels to be scored
        elif ratio >= 70: # lower confidence, still considered a "hit" for gating sub-levels
            achieved_score += weights['WorkTitle'] * (ratio / 100.0) * 0.5 # penalized score
            work_title_matched_sufficiently = True
        if VERBOSE_DEBUG_SCORING: print(f"SCORE_DEBUG: WorkTitle GT='{gt_work_title_val}', Pred='{pred_work_title_val}', Ratio={ratio}, CurrentAchievedScore={achieved_score:.2f}")
    
    # score Genre
    gt_genre_val = gt_universal_dict.get('Genre')
    pred_genre_val = pred_universal_dict.get('Genre')
    if gt_genre_val and pred_genre_val:
        # score genre somewhat independently, but it's more meaningful if WorkTitle was close
        base_genre_weight = weights['Genre']
        if not work_title_matched_sufficiently:
             base_genre_weight *= 0.5 # Reduce impact if work title was off

        ratio = fuzz.token_set_ratio(pred_genre_val, gt_genre_val)
        if ratio >= 80:
            achieved_score += base_genre_weight * (ratio / 100.0)
        if VERBOSE_DEBUG_SCORING: print(f"SCORE_DEBUG: Genre GT='{gt_genre_val}', Pred='{pred_genre_val}', Ratio={ratio}, CurrentAchievedScore={achieved_score:.2f}")

    # score Hierarchical Levels (Book, Chapter, Verse) - Soft Gating
    if work_title_matched_sufficiently: # only proceed if WorkTitle was at least somewhat identified
        for label in ['Book', 'Chapter', 'Verse']:
            gt_level_val = gt_universal_dict.get(label)
            pred_level_val = pred_universal_dict.get(label)

            if gt_level_val is not None: # This level is expected by GT
                if pred_level_val is not None and pred_level_val.lower() != "unknown":
                    ratio = fuzz.token_set_ratio(pred_level_val, gt_level_val)
                    if ratio >= 80: # Good match for this level's descriptive value
                        achieved_score += weights[label] * (ratio / 100.0)
                    elif ratio >= 60 : # Partial credit for somewhat similar
                         achieved_score += weights[label] * (ratio / 100.0) * 0.5 
                    if VERBOSE_DEBUG_SCORING: print(f"SCORE_DEBUG: Level '{label}' GT='{gt_level_val}', Pred='{pred_level_val}', Ratio={ratio}, CurrentAchievedScore={achieved_score:.2f}")
                elif VERBOSE_DEBUG_SCORING: 
                    print(f"SCORE_DEBUG: Level '{label}' GT='{gt_level_val}', Pred=None/Unknown. LLM missed this expected level.")
            elif pred_level_val is not None and pred_level_val.lower() != "unknown": # GT didn't expect, LLM provided
                achieved_score -= weights[label] * 0.25 # Penalty for hallucinating an inapplicable level
                if VERBOSE_DEBUG_SCORING: print(f"SCORE_DEBUG: Penalty for hallucinated level '{label}': {pred_level_val}. ScoreNow={achieved_score:.2f}")
    
    if max_possible_score_for_sample == 0:
        # This case means GT had no universal fields populated (should not happen with transformed data)
        # If prediction is also all None/Unknown, it's a "correct empty match".
        all_pred_empty_or_unknown = all(
            (pred_universal_dict.get(lbl) is None or pred_universal_dict.get(lbl).lower() == "unknown")
            for lbl in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER if weights.get(lbl, 0) > 0
        )
        return 1.0 if all_pred_empty_or_unknown else 0.0

    final_reward = achieved_score / max_possible_score_for_sample if max_possible_score_for_sample > 0 else 0.0
    return max(0.0, min(1.0, final_reward))


def compute_sanskrit_source_retrieval_reward(llm_raw_answer: str, ground_truth_source_of_origin: dict) -> float:
    """
    Calculates the reward. Assumes ground_truth_source_of_origin *already contains*
    the universal fields: WorkTitle, Genre, Book, Chapter, Verse.
    """
    global VERBOSE_DEBUG_SCORING # Allow global override for detailed debugging
    
    if not llm_raw_answer or not isinstance(llm_raw_answer, str):
        if VERBOSE_DEBUG_SCORING: print("REWARD_FUNC: LLM raw answer is invalid.")
        return 0.0
    if not ground_truth_source_of_origin or not isinstance(ground_truth_source_of_origin, dict):
        if VERBOSE_DEBUG_SCORING: print("REWARD_FUNC: Ground truth source_of_origin is invalid.")
        return 0.0

    # Extract the pre-transformed universal fields from GT
    gt_universal = {
        label: ground_truth_source_of_origin.get(label) 
        for label in SOURCE_RETRIEVAL_UNIVERSAL_LABELS_ORDER
    }
    # Filter out None values from GT, as these don't contribute to max_possible_score
    gt_universal_filtered = {k: v for k, v in gt_universal.items() if v is not None}
    if not gt_universal_filtered.get("WorkTitle"): # If GT itself is missing a WorkTitle after transformation
        if VERBOSE_DEBUG_SCORING: print("REWARD_FUNC: CRITICAL - GT universal is missing WorkTitle.")
        return 0.0


    pred_universal = parse_llm_answer_to_universal(llm_raw_answer)

    if VERBOSE_DEBUG_SCORING:
        print(f"--- Scoring ---")
        print(f"GT Universal (from dataset): {json.dumps(gt_universal_filtered, ensure_ascii=False, indent=2)}")
        print(f"LLM Raw Answer: {llm_raw_answer}")
        print(f"LLM Parsed Universal: {json.dumps(pred_universal, ensure_ascii=False, indent=2)}")

    score = llm_scoring(pred_universal, gt_universal_filtered)
    if VERBOSE_DEBUG_SCORING: print(f"Final Score: {score:.4f}\n-----------")
    
    return score
