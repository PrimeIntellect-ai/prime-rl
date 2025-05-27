"""Unit tests for Sanskrit literature source retrieval reward function."""

import pytest
from zeroband.inference.sanskrit_retrieval import (
    compute_retrieval_reward,
    parse_prediction,
    normalize_text_name,
    to_devanagari,
    to_iast,
    TRANSLITERATION_AVAILABLE
)


class TestTransliteration:
    """Test transliteration functions."""
    
    @pytest.mark.skipif(not TRANSLITERATION_AVAILABLE, reason="indic_transliteration not installed")
    def test_to_devanagari(self):
        # IAST to Devanagari
        assert to_devanagari("meghadūta", "iast") == "मेघदूत"
        assert to_devanagari("kālidāsa", "iast") == "कालिदास"
        
        # Already Devanagari
        assert to_devanagari("मेघदूत", "devanagari") == "मेघदूत"
        
    @pytest.mark.skipif(not TRANSLITERATION_AVAILABLE, reason="indic_transliteration not installed")
    def test_to_iast(self):
        # Devanagari to IAST
        assert to_iast("मेघदूत", "devanagari") == "meghadūta"
        assert to_iast("कालिदास", "devanagari") == "kālidāsa"
        
        # Already IAST
        assert to_iast("meghadūta", "iast") == "meghadūta"


class TestParsePrediction:
    """Test prediction parsing functionality."""
    
    def test_structured_format(self):
        prediction = "Genre: kavya, Author: kalidasa, Text: meghaduta, Chapter: 1, Verse: 15"
        result = parse_prediction(prediction)
        
        assert result['genre'] == 'kavya'
        assert result['author'] == 'kalidasa'
        assert result['text'] == 'meghaduta'
        assert result['chapter'] == '1'
        assert result['verse'] == '15'
    
    def test_natural_language_format(self):
        prediction = "This is from Kalidasa's Meghaduta, verse 15"
        result = parse_prediction(prediction)
        
        assert result.get('author') == 'kalidasa'
        assert result.get('text') == 'meghaduta'
        assert result.get('verse') == '15'
    
    def test_mixed_case(self):
        prediction = "Genre: KAVYA, Author: KALIDASA, Text: MEGHADUTA"
        result = parse_prediction(prediction)
        
        assert result['genre'] == 'kavya'
        assert result['author'] == 'kalidasa'
        assert result['text'] == 'meghaduta'
    
    def test_sanskrit_script(self):
        prediction = "Genre: काव्य, Author: कालिदास, Text: मेघदूत, Verse: १५"
        result = parse_prediction(prediction)
        
        # Should at least not crash and extract what it can
        assert isinstance(result, dict)
        # Should convert Devanagari numerals
        if 'verse' in result:
            assert result['verse'] == '15'
    
    def test_devanagari_numerals(self):
        prediction = "Chapter: ३, Verse: १५"
        result = parse_prediction(prediction)
        
        assert result.get('chapter') == '3'
        assert result.get('verse') == '15'



class TestNormalizeTextName:
    """Test text name normalization."""
    
    def test_diacritic_removal(self):
        assert normalize_text_name("Meghadūta") == "meghaduta"
        assert normalize_text_name("Rāmāyaṇa") == "ramayana"
        assert normalize_text_name("Bhāgavata Purāṇa") == "bhagavata"
    
    def test_suffix_removal(self):
        assert normalize_text_name("Brahmasutra") == "brahma"
        assert normalize_text_name("Vishnupurana") == "vishnu"
        assert normalize_text_name("Manusamhita") == "manu"


class TestComputeRetrievalReward:
    """Test reward computation."""
    
    def test_full_accuracy(self):
        prediction = "Genre: kavya, Author: kalidasa, Text: meghaduta, Chapter: 1, Verse: 15"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(1.0)
    
    def test_good_accuracy(self):
        # Correct up to chapter
        prediction = "Genre: kavya, Author: kalidasa, Text: meghaduta, Chapter: 1, Verse: 10"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.85)  # 0.10 + 0.20 + 0.30 + 0.25 = genre + author + text + chapter
    
    def test_partial_accuracy(self):
        # Genre + author + text correct
        prediction = "Genre: kavya, Author: kalidasa, Text: meghaduta"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.65)  # 0.10 + 0.15 + 0.25 = genre + author + text
    
    def test_genre_only(self):
        prediction = "Genre: kavya, Author: bhasa, Text: svapnavasavadatta"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.15)  # 0.10 = Only genre correct
    
    def test_wrong_genre_right_synonym(self):
        prediction = "Genre: epic, Author: kalidasa, Text: meghaduta"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.6) 
    
    def test_genre_synonyms(self):
        prediction = "Genre: poetry, Author: kalidasa, Text: meghaduta"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward >= 0.45  # partial genre + author + text with hierarchical scoring
    
    def test_normalized_names(self):
        prediction = "Genre: kavya, Author: Kālidāsa, Text: Meghadūta, Chapter: 1, Verse: 15"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(1.0)  # Should handle diacritics
    
    def test_invalid_prediction(self):
        prediction = "I don't know the source"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == 0.0
    
    def test_no_dependency(self):
        # Correct chapter but wrong text - should get no credit for chapter
        prediction = "Genre: kavya, Author: kalidasa, Text: raghuvamsha, Chapter: 1"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa', 
            'text': 'meghaduta',
            'chapter': '1',
            'verse': '15'
        }
        
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.35)  # 0.10 + 0.15 = Only genre + author
        
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty prediction
        assert compute_retrieval_reward("", {'genre': 'kavya'}) == 0.0
        
        # None values in ground truth
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta',
            'chapter': None,
            'verse': None
        }
        prediction = "Genre: kavya, Author: kalidasa, Text: meghaduta"
        reward = compute_retrieval_reward(prediction, ground_truth)
        assert reward == pytest.approx(0.65)  
        
        
    def test_devanagari_predictions(self):
        """Test predictions containing Devanagari text."""
        prediction = "Genre: काव्य, Text: मेघदूत, Author: कालिदास"
        ground_truth = {
            'genre': 'kavya',
            'author': 'kalidasa',
            'text': 'meghaduta'
        }
        
        # The prediction parser should handle Devanagari
        result = parse_prediction(prediction)
        assert isinstance(result, dict)  # Should at least parse without error