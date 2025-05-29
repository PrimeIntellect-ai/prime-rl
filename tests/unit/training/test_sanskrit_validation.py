"""Unit tests for Sanskrit validation and anti-gaming features."""

import pytest
from zeroband.inference.sanskrit_retrieval import (
    has_sanskrit_content,
    normalize_text_name,
    compute_retrieval_reward
)


class TestDiacriticNormalization:
    """Test handling of missing diacritics."""
    
    def test_normalize_without_diacritics(self):
        """Test that normalization handles missing diacritics."""
        # Model output without diacritics
        assert normalize_text_name("Kalidasa") == "kalidasa"
        assert normalize_text_name("Ramayana") == "ramayana"
        
        # Ground truth with diacritics
        assert normalize_text_name("Kālidāsa") == "kalidasa"
        assert normalize_text_name("Rāmāyaṇa") == "ramayana"
        
        # Both should match after normalization
        assert normalize_text_name("Kalidasa") == normalize_text_name("Kālidāsa")
        
    def test_reward_with_missing_diacritics(self):
        """Test that rewards work even when model doesn't output diacritics."""
        # Model prediction without diacritics
        prediction = "Genre: kavya, Author: Kalidasa, Text: Meghaduta, Verse: 1"
        
        # Ground truth with diacritics
        ground_truth = {
            'genre': 'kāvya',
            'author': 'Kālidāsa',
            'text': 'Meghadūta',
            'verse': '1'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        # Should still get points for correct identification
        assert reward >= 0.50  # 0.10 + 0.15 + 0.25 = genre + author + text


class TestSanskritContentDetection:
    """Test Sanskrit content detection."""
    
    def test_devanagari_detection(self):
        """Test Devanagari script detection."""
        assert has_sanskrit_content("धर्मक्षेत्रे कुरुक्षेत्रे") == True
        assert has_sanskrit_content("यदा यदा हि धर्मस्य") == True
        assert has_sanskrit_content("Only English text") == False
        
    def test_iast_pattern_detection(self):
        """Test IAST pattern detection."""
        assert has_sanskrit_content("dharmakṣetre kurukṣetre") == True
        assert has_sanskrit_content("yadā yadā hi dharmasya") == True
        
        # Common Sanskrit terms
        assert has_sanskrit_content("This is from the Mahabharata") == True
        assert has_sanskrit_content("This talks about dharma and karma") == True
        
    def test_mixed_content(self):
        """Test mixed Sanskrit and English."""
        assert has_sanskrit_content("The verse यदा यदा हि धर्मस्य is famous") == True
        assert has_sanskrit_content("From Kalidasa's work on kavya") == True


class TestRobustParsing:
    """Test robust parsing of various answer formats."""
    
    def test_informal_answers(self):
        """Test parsing of informal/conversational answers."""
        predictions = [
            "I think this is from Kalidasa's Meghaduta",
            "This looks like it's from the Bhagavad Gita",
            "Could be from Ramayana, maybe chapter 2",
            "Probably Mahabharata, not sure which verse"
        ]
        
        for pred in predictions:
            # Should not crash and should extract some information
            ground_truth = {'genre': 'epic', 'text': 'test'}
            reward = compute_retrieval_reward(pred, ground_truth)
            assert isinstance(reward, float)

        

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_predictions(self):
        """Test handling of empty or minimal predictions."""
        ground_truth = {'genre': 'kavya', 'text': 'test'}
        
        assert compute_retrieval_reward("", ground_truth) == 0.0
        assert compute_retrieval_reward("   ", ground_truth) == 0.0
        assert compute_retrieval_reward("???", ground_truth) == 0.0
        
        
    def test_numerical_variations(self):
        """Test handling of numerical variations."""
        predictions = [
            "Chapter: 1, Verse: 15",
            "Chapter: १, Verse: १५",  # Devanagari numerals
            "Chapter: I, Verse: XV",   # Roman numerals
            "Chapter 1 Verse 15",      # No punctuation
        ]
        
        ground_truth = {'chapter': '1', 'verse': '15', 'genre': 'epic'}
        
        # First two should work
        for pred in predictions[:2]:
            full_pred = f"Genre: epic, {pred}"
            reward = compute_retrieval_reward(full_pred, ground_truth)
            assert reward >= 0.10  # At least genre match (0.10)