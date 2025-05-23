# tests/test_sanskrit_poetry_chandas.py
import pytest
from zeroband.inference.genesys.sanskrit import compute_sanskrit_poetry_reward

def test_sloka_meter_identification():
    """Test with actual Sanskrit sloka."""
    # From Bhagavad Gita
    completion = """<think>I need to compose in Śloka meter</think>
धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"""
    
    verification_info = {"meter_type": "Śloka", "topic": "dharma"}
    reward = compute_sanskrit_poetry_reward(completion, verification_info)
    assert reward > 0.5  # Should recognize as Śloka

def test_anti_gaming():
    """Test that repetitive patterns are rejected."""
    completion = """<think>Gaming the system</think>
ल ल ल ल ल ल ल ल
ल ल ल ल ल ल ल ल"""
    
    verification_info = {"meter_type": "Śloka", "topic": "test"}
    reward = compute_sanskrit_poetry_reward(completion, verification_info)
    assert reward == 0.0  # Should reject repetitive pattern

def test_vasantatilaka_meter():
    """Test Vasantatilakā meter recognition."""
    # From Kalidasa
    completion = """<think>Composing in Vasantatilakā</think>
कश्चित्कान्ताविरहगुरुणा स्वाधिकारात्प्रमत्तः
शापेनास्तंगमितमहिमा वर्षभोग्येण भर्तुः"""
    
    verification_info = {"meter_type": "Vasantatilakā", "topic": "love"}
    reward = compute_sanskrit_poetry_reward(completion, verification_info)
    assert reward > 0.5