# Update src/zeroband/inference/genesys/sanskrit.py
from typing import Dict
def compute_sanskrit_poetry_reward(completion: str, verification_info: Dict) -> float:
    """
    Enhanced reward function with anti-gaming measures.
    """
    # Extract poem
    if "</think>" in completion:
        poem = completion.split("</think>")[1].strip()
    else:
        return 0.0
    
    # Basic checks to prevent gaming
    if len(poem) < 20:  # Too short
        return 0.0
    
    # Check for repetitive patterns (anti-gaming)
    if is_repetitive(poem):
        return 0.0
    
    # These should always be present - fail fast if missing
    expected_meter = verification_info["meter_type"]
    topic = verification_info["topic"]
    
    # Verify meter using Chandas
    meter_score = verify_meter_with_chandas(poem, expected_meter)
    
    # Optional: LLM grading for content quality
    # content_score, _ = llm_grader.grade_poetry(poem, topic, expected_meter)
    
    # For now, just use meter score
    return meter_score

def is_repetitive(poem: str) -> bool:
    """Check if poem is just repetitive syllables."""
    words = poem.split()
    if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
        return True
    
    # Check for repeated short patterns
    for i in range(2, 6):
        pattern = poem[:i]
        if poem.count(pattern) > len(poem) // (i * 2):
            return True
    
    return False

def verify_meter_with_chandas(poem: str, expected_meter: str) -> float:
    """Use Chandas to verify meter."""
    try:
        lines = poem.strip().split('\n')
        pattern_lines = identify.to_pattern_lines(lines)
        id_result = identify.identifier.IdentifyFromPatternLines(pattern_lines)
        
        expected_norm = normalize_meter_name(expected_meter)
        
        # Explicit checks instead of .get() - be clear about what we expect
        if 'exact' in id_result and id_result['exact']:
            if normalize_meter_name(id_result['exact']) == expected_norm:
                return 1.0
        
        # Check partial matches only if exact match failed
        if 'partial' in id_result and id_result['partial']:
            for partial in id_result['partial']:
                if normalize_meter_name(partial) == expected_norm:
                    return 0.7
        
        return 0.0
        
    except Exception as e:
        print(f"Chandas error: {e}")
        return 0.0