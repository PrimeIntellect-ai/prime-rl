"""
Sanskrit morphology reward computation using Paninian grammar rules.

Call compute_sanskrit_morphology_reward(completion: str, verification_info: Dict).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import re

from vidyut.prakriya import Dhatu, Gana, Lakara, Prayoga, Purusha, Vacana, Pada, Vyakarana, Prakriya
from vidyut.lipi import Scheme, transliterate, detect


# --- Main Reward Function ---
def compute_sanskrit_morphology_reward(completion: str, verification_info: Dict) -> float:
    """
    Compute reward for Sanskrit morphology generation task.
    
    Args:
        completion: The model's generated Sanskrit surface form
        verification_info: Dictionary containing morphological specification
    
    Returns:
        float: Reward score between 1.0 and 0.0
    """
    try:
        # Extract the generated form from completion
        generated_form = _extract_answer(completion)

        # Convert to SLP1 scheme
        generated_form = _convert_to_slp1(generated_form)
        if not generated_form:
            return 0.0
        
        # Get all possible derivations using Vidyut-Prakriya
        derivations = _paninian_surface_form_derivations(verification_info)
        
        # Verify the generated form against derivations
        return _verify_generated_form_with_score(generated_form, derivations)
        
    except Exception:
        return 0.0


# --- Text Processing Utilities ---
def _extract_answer(completion: str) -> Optional[str]:
    """
    Extract the Sanskrit form from the model's completion. Expects the answer to be in [[surface_form]] format.
    
    Args:
        completion: Full model output text
        
    Returns:
        Optional[str]: Extracted Sanskrit form, or None if not found or invalid
    """
    matches = re.findall(r'\[\[([^\]]*)\]\]', completion)
    content = matches[-1].strip() if matches else ""
    return content if content and ' ' not in content else None


def _convert_to_slp1(sanskrit_text: str) -> Optional[str]:
    """
    Convert the given text to SLP1 (Sanskrit Language Processing 1) romanization.
    
    Args:
        sanskrit_text: The text to convert
    
    Returns:
        Optional[str]: The converted text in SLP1 romanization, or None if detection fails
    """
    if not sanskrit_text:
        return None
    
    input_scheme = detect(sanskrit_text)
    if input_scheme is None:
        return None
    
    return transliterate(sanskrit_text, input_scheme, Scheme.Slp1)


# --- Paninian Grammar ---
def _paninian_surface_form_derivations(verification_info: Dict) -> List[Prakriya]:
    """
    Generate all possible Sanskrit surface forms using Vidyut-Prakriya.
    NB: Vidyut uses SLP1 romanization internally.
    
    Args:
        verification_info: Dictionary containing morphological specification
        
    Returns:
        List[Prakriya]: List of derivation objects containing surface forms and derivation steps
    """
    spec = VerbMorphologySpecification.from_dict(verification_info)
    v = Vyakarana()
    
    mula_dhatu = Dhatu.mula(aupadeshika=spec.dhatu, gana=spec.gana)
    prakriyas = v.derive(Pada.Tinanta(
        dhatu=mula_dhatu,
        prayoga=spec.prayoga,
        lakara=spec.lakara,
        purusha=spec.purusha,
        vacana=spec.vacana,
    ))
    
    # Return all derivations
    return prakriyas


def _verify_generated_form_with_score(generated_form: str, prakriyas: List[Prakriya]) -> float:
    """
    Verify if the generated form matches any of the derivations.
    
    Args:
        generated_form: The generated Sanskrit form in SLP1 romanization
        prakriyas: List of possible derivations
    
    Returns:
        float: Correctness score (1.0 for exact match, scaled partial score for intermediate form matches)
    """
    # Check for exact match with correct forms
    correct_forms = {prakriya.text for prakriya in prakriyas}
    if generated_form in correct_forms:
        return 1.0

    # Check for partial correctness (intermediate forms in derivation steps)
    max_score = 0.0
    for prakriya in prakriyas:
        steps = prakriya.history or []
        for i, step in enumerate(steps):
            intermediate_form = ''.join(step.result)
            if generated_form == intermediate_form:
                score = (i + 1) / len(steps)
                max_score = max(max_score, score)
    
    return max_score


# --- Data Class ---
@dataclass
class VerbMorphologySpecification:
    """Contains all the grammatical parameters to specify a tinanta pada (finite verb surface form)"""
    dhatu: str          # Root verb (aupadeshika form)
    gana: Gana          # Verb class 
    lakara: Lakara      # Tense/mood
    prayoga: Prayoga    # Voice 
    purusha: Purusha    # Person 
    vacana: Vacana      # Number 
    
    def to_dict(self) -> dict:
        return {k: str(v) for k, v in vars(self).items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> "VerbMorphologySpecification":
        return cls(
            dhatu=d["dhatu"],
            gana=Gana(d["gana"]),
            lakara=Lakara(d["lakara"]),
            prayoga=Prayoga(d["prayoga"]),
            purusha=Purusha(d["purusha"]),
            vacana=Vacana(d["vacana"]),
        )
