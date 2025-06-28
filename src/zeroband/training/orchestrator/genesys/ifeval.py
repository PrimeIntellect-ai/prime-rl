# adapted from https://github.com/allenai/open-instruct/blob/main/scripts/eval_constraints/if_functions.py
import json
import re
from typing import List


def verify_keywords(text, keyword_list):
    """
    Verify if the response contains all the specified keywords.

    Args:
        response (str): The response text to check
        keyword_list (list): A list of keywords to check for

    Returns:
        bool: True if all keywords are present in the response, False otherwise
    """
    # Convert response to lowercase for case-insensitive matching
    response_lower = text.lower()

    # Check if all keywords are present in the response
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N):
    """
    Verifies if a keyword appears exactly N times in the given text.

    Args:
        text (str): The text to analyze
        keyword (str): The keyword to count
        expected_count (int): The expected number of occurrences

    Returns:
        tuple: (bool, int) - (Whether constraint is met, actual count found)
    """
    # Convert text to lowercase to make the search case-insensitive
    text = text.lower()
    keyword = word.lower()

    # Split text into words and remove punctuation
    import re

    words = re.findall(r"\b\w+\b", text)

    # Count actual occurrences
    actual_count = sum(1 for word in words if word == keyword)

    # Check if constraint is met
    constraint_met = actual_count == N

    return constraint_met


def validate_forbidden_words(text, forbidden_words):
    """
    Validates that the text does not contain any of the specified forbidden words.

    Args:
        text (str): The text to check for forbidden words
        forbidden_words (list[str]): A list of forbidden words

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - Boolean indicating if any forbidden words are present
            - List of forbidden words found in the text

    Example:
        text = "This is a message that should not contain any bad words"
        forbidden_words = ["bad", "evil", "harmful"]
        result = validate_forbidden_words(text, forbidden_words)
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Check each forbidden word
    found_words = [word for word in forbidden_words if word.lower() in text_lower]

    # Return results
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """
    Verifies if a given letter appears exactly the specified number of times in the text.

    Args:
        text (str): The text to check
        letter (str): The letter to count (case-sensitive)
        target_count (int): The expected number of occurrences

    Returns:
        bool: True if the constraint is met, False otherwise

    Example:
        >>> verify_letter_frequency("hello world", "l", 3)
        True
        >>> verify_letter_frequency("hello world", "o", 2)
        True
        >>> verify_letter_frequency("hello world", "x", 0)
        True
    """
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")

    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text, language):
    """
    Validates that the entire response is in the specified language.

    Args:
        text (str): The text to check
        language (str): The language code (e.g., 'en' for English)

    Returns:
        bool: True if the response is entirely in the specified language, False otherwise

    Example:
        text = "This is an English sentence"
        language = "en"
        result = validate_response_language(text, language)
    """
    from langdetect import detect

    # Detect the language of the text
    detected_language = detect(text)
    # Check if the detected language matches the expected language
    return detected_language == language


def verify_paragraph_count(text: str, N: int) -> bool:
    """
    Verifies that a text contains the expected number of paragraphs,
    where paragraphs are separated by markdown dividers '* * *'

    Args:
        text (str): The text to analyze
        expected_count (int): Expected number of paragraphs

    Returns:
        bool: True if the text contains exactly the expected number of paragraphs,
              False otherwise

    Example:
         text = "First paragraph\n* * *\nSecond paragraph"
         verify_paragraph_count(text, 2)
        True
    """

    def clean_text(text: str) -> str:
        """Remove extra whitespace and normalize line endings"""
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    # Clean the input text
    text = clean_text(text)

    # Split text by markdown divider
    # Add 1 to count since n dividers create n+1 paragraphs
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)

    # Verify each split resulted in non-empty content
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False

    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Validates if a text meets specified word count constraints.

    Args:
        text (str): The text to check
        count (int): The target word count
        qualifier (str): The type of constraint ('at least', 'around', 'at most')

    Returns:
        bool: True if the constraint is met, False otherwise

    Raises:
        ValueError: If an invalid qualifier is provided
    """
    # Remove extra whitespace and split into words
    words = text.strip().split()
    actual_count = len(words)

    # Define tolerance for "around" qualifier (±10% of target count)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Verifies if a text contains the expected number of sentences.

    Args:
        text (str): The text to analyze
        N (int): The expected number of sentences
        quantifier (str): The quantifier ('at least', 'around', 'at most')

    Returns:
        bool: True if the text contains the expected number of sentences, False otherwise
    """
    # Split the text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Count the number of sentences
    actual_count = len(sentences)

    # Check if the actual count matches the expected count based on the quantifier
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


def validate_paragraphs(text, N, first_word, i):
    """
    Validates that a text contains the expected number of paragraphs and that the i-th paragraph starts with a specific
    word.

    Args:
        text (str): The text to analyze
        N (int): The expected number of paragraphs
        first_word (str): The expected first word of the i-th paragraph
        i (int): The index of the paragraph to check (1-indexed)

    Returns:
        bool: True if the text meets the paragraph and first word requirements, False otherwise
    """
    # Split the text into paragraphs
    paragraphs = text.split("\n\n")

    # Check if the number of paragraphs is as expected
    if len(paragraphs) != N:
        return False

    # Check if the i-th paragraph starts with the specified first word
    if paragraphs[i - 1].strip().startswith(first_word):
        return True
    return False


def verify_postscript(text, postscript_marker):
    """
    Verifies if a text contains a postscript starting with '{postscript marker}'

    Args:
        text (str): The text to verify

    Returns:
        bool: True if the text contains a valid postscript, False otherwise
    """
    # Check if the text contains the postscript marker
    if postscript_marker in text:
        # Get the index of the marker
        marker_index = text.find(postscript_marker)
        # Check if the marker appears near the end
        remaining_text = text[marker_index:].strip()
        # Verify it's not just the marker alone
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> tuple[bool, List[str]]:
    """
    Validates if a text contains at least the specified number of placeholders in square brackets.

    Args:
        text (str): The text to check for placeholders
        min_placeholders (int): Minimum number of placeholders required

    Returns:
        tuple[bool, List[str]]: A tuple containing:
            - Boolean indicating if the text meets the placeholder requirement
            - List of found placeholders

    Example:
        >>> text = "Hello [name], your [item] will be delivered to [address]"
        >>> validate_placeholders(text, 2)
        (True, ['name', 'item', 'address'])
    """
    # Find all placeholders using regex
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)

    # Check if the number of placeholders meets the requirement
    has_enough = len(placeholders) >= N

    return has_enough


def verify_bullet_points(text: str, N: int) -> tuple[bool, str]:
    """
    Verifies if a text contains exactly N bullet points in markdown format.
    Returns a tuple of (is_valid, message).

    Args:
        text (str): The text to check
        expected_count (int): The expected number of bullet points

    Returns:
        tuple[bool, str]: (True if constraint is met, explanation message)
    """
    # Split text into lines and count lines starting with * or -
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)

    if actual_count == N:
        return True
    else:
        return False


def validate_title(text: str) -> bool:
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)

    if len(matches) > 0:
        return True
    else:
        return False


def validate_choice(text: str, options: list) -> bool:
    for option in options:
        if text in option:
            return True
    return False


def validate_highlighted_sections(text: str, N: int) -> bool:
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)

    if len(matches) >= N:
        return True
    else:
        return False


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    sections = text.split(section_splitter)
    # The first section might not start with the splitter, so we adjust for this
    if sections[0] == "":
        sections.pop(0)
    if len(sections) == N:
        return True
    else:
        return False


def validate_json_format(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    if text.startswith(original_prompt):
        return True
    else:
        return False


def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


def validate_uppercase(text: str) -> bool:
    # Check if the response is the same as the uppercase version of the response
    if text == text.upper():
        return True
    else:
        return False


def validate_lowercase(text: str) -> bool:
    # Check if the response is the same as the lowercase version of the response
    if text == text.lower():
        return True
    else:
        return False


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


def validate_end(text: str, end_phrase: str) -> bool:
    # Check if the response ends with the end phrase
    if text.endswith(end_phrase):
        return True
    else:
        return False


def validate_quotation(text: str) -> bool:
    if text.startswith('"') and text.endswith('"'):
        return True
    else:
        return False


def validate_no_commas(text: str) -> bool:
    if "," not in text:
        return True
    else:
        return False


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}


def verify_ifeval(completion: str, verification_info: dict) -> float:
    """
    Verifies completion against IFEval instruction-following criteria.

    Args:
        completion: Model's response string
        verification_info: Dict containing ground_truth with func_name and parameters

    Returns:
        Float score (0.0 to 1.0) representing instruction following accuracy
    """

    # Extract the response after thinking (if present)
    if "</think>" in completion:
        response = completion.split("</think>")[1].strip()
    else:
        response = completion.strip()

    if not response:
        return 0.0

    try:
        # Get ground truth from verification_info
        ground_truth = verification_info.get("ground_truth")
        if isinstance(ground_truth, str):
            gt = json.loads(ground_truth)
        else:
            gt = ground_truth

        if not gt:
            return 0.0

        # Extract function name and parameters
        func_name = gt.pop("func_name")
        func = IF_FUNCTIONS_MAP.get(func_name)

        if not func:
            return 0.0

        # Filter out None values and pass to function
        non_none_args = {k: v for k, v in gt.items() if v is not None}

        # Call the verification function
        result = func(response, **non_none_args)

        # Convert boolean or other results to float score
        if isinstance(result, bool):
            return float(result)
        elif isinstance(result, (int, float)):
            return float(result)
        elif isinstance(result, tuple):
            # Some functions return tuples, use first element if boolean
            if len(result) > 0 and isinstance(result[0], bool):
                return float(result[0])
            return 0.0
        else:
            return 0.0

    except Exception:
        return 0.0
