# Update src/zeroband/inference/genesys/sanskrit.py
def compute_sanskrit_poetry_reward(completion: str, verification_info: Dict) -> float:
    """
    Enhanced reward function with anti-gaming measures.
    """
    if not CHANDAS_AVAILABLE:
        return 0.0
    
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
    
    expected_meter = verification_info.get("meter_type", "Śloka")
    topic = verification_info.get("topic", "")
    
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
        
        # Exact match
        if id_result.get('exact'):
            if normalize_meter_name(id_result['exact']) == expected_norm:
                return 1.0
        
        # Partial match
        if id_result.get('partial'):
            for partial in id_result['partial']:
                if normalize_meter_name(partial) == expected_norm:
                    return 0.7
        
        return 0.0
        
    except Exception as e:
        print(f"Chandas error: {e}")
        return 0.0